"""
train_gemma.py
Script for fine-tuning Gemma-2B, supporting distributed training, gradient accumulation, early stopping, etc.
"""
import os
import argparse
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def setup_ddp():
    """Initialize distributed training environment"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_ddp():
    """Clean up distributed training environment"""
    dist.destroy_process_group()


def is_main_process():
    """Check if the current process is the main process"""
    return not dist.is_initialized() or dist.get_rank() == 0


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CausalLMDataset(Dataset):
    """
    Dataset for causal language modeling.
    Supports pure text (one line per file) or JSONL (with 'text' field).
    """
    def __init__(self, file_path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path.resolve()}")
        with path.open('r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        self.encodings = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                text = obj.get('text', '') or ln
            except json.JSONDecodeError:
                text = ln
            enc = tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            input_ids = enc.input_ids.squeeze()
            attention_mask = enc.attention_mask.squeeze()
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            self.encodings.append({
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, args, epoch, tokenizer):
    model.train()
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"训练中 (Epoch {epoch+1})", disable=not is_main_process())
    for step, batch in enumerate(progress):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.autocast(device_type=device.type,
                              dtype=torch.bfloat16 if device.type=='cuda' else torch.float32,
                              enabled=(device.type=='cuda')):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            if args.gradient_accumulation_steps>1:
                loss = loss/args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if is_main_process() and (step+1)%200==0:
            model.eval()
            with torch.no_grad():
                gen_model = model.module if hasattr(model, 'module') else model
                generated = gen_model.generate(input_ids=input_ids[:1], max_new_tokens=50, do_sample=False)
            decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            decoded_pred  = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"\nStep {step+1} Input: {decoded_input}")
            print(f"Step {step+1} Predicted Output: {decoded_pred}\n")
            model.train()

        if (step+1)%args.gradient_accumulation_steps==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()*args.gradient_accumulation_steps
        if is_main_process():
            progress.set_postfix({'loss': total_loss/(step+1), 'lr': scheduler.get_last_lr()[0]})
    return total_loss/len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss=0.0; batches=0
    progress=tqdm(dataloader, desc="Validation", disable=not is_main_process())
    with torch.no_grad():
        for batch in progress:
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)
            outputs=model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss+=outputs.loss.item(); batches+=1
            if is_main_process(): progress.set_postfix({'loss': total_loss/batches})
    if dist.is_initialized():
        stats=torch.tensor([total_loss,batches]).to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss,batches=stats.tolist()
    return total_loss/batches if batches>0 else 0


def main(args):
    is_ddp='WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE'])>1
    if is_ddp: setup_ddp()
    set_seed(args.seed)
    device=torch.device(f"cuda:{os.environ['LOCAL_RANK']}" if is_ddp else ("cuda" if torch.cuda.is_available() else "cpu"))
    if is_main_process(): print(f"Distributed: {is_ddp}, Device: {device}"), print(f"Loading model/tokenizer: {args.model_name}")
    tokenizer=AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model=AutoModelForCausalLM.from_pretrained(args.model_name)
    model.gradient_checkpointing_enable(); model.to(device)
    if is_ddp: model=DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False)
    if args.use_torch_compile and hasattr(torch,'compile') and is_main_process(): model=torch.compile(model)
    if is_main_process(): print("Loading datasets...")
    train_ds=CausalLMDataset(args.train_file, tokenizer, args.max_input_length)
    val_ds  =CausalLMDataset(args.validation_file, tokenizer, args.max_input_length)
    train_dl=DataLoader(train_ds,batch_size=args.batch_size,sampler=DistributedSampler(train_ds) if is_ddp else None, shuffle=not is_ddp, num_workers=args.num_workers,pin_memory=True)
    val_dl  =DataLoader(val_ds  ,batch_size=args.batch_size*2,sampler=DistributedSampler(val_ds)   if is_ddp else None, shuffle=False     , num_workers=args.num_workers,pin_memory=True)
    optimizer=AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps=args.epochs*(len(train_dl)//args.gradient_accumulation_steps)
    scheduler=get_scheduler(name=args.scheduler_type, optimizer=optimizer, num_warmup_steps=int(total_steps*args.warmup_ratio), num_training_steps=total_steps)
    scaler=torch.amp.GradScaler(enabled=device.type=='cuda')
    best_loss=float('inf'); epochs_no_improve=0
    if is_main_process(): Path(args.output_dir).mkdir(parents=True,exist_ok=True); print("Starting training...")
    for epoch in range(args.epochs):
        if is_main_process(): print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss=train_one_epoch(model, train_dl, optimizer, scheduler, scaler, device, args, epoch, tokenizer)
        if is_ddp: dist.barrier()
        if is_main_process(): print(f"Epoch {epoch+1} train loss: {train_loss:.4f}")
        val_loss=evaluate(model, val_dl, device)
        if is_main_process():
            print(f"Val loss: {val_loss:.4f}")
            epoch_dir = f"{args.output_dir}/epoch_{epoch+1}"
            Path(epoch_dir).mkdir(parents=True, exist_ok=True)
            unwrapped = model.module if hasattr(model, 'module') else model
            if hasattr(unwrapped, '_orig_mod'): unwrapped = unwrapped._orig_mod
            unwrapped.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)
            print(f"Saved model for epoch {epoch+1} to {epoch_dir}")    
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                unwrapped.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"Saved new best model (loss={best_loss:.4f}) to {args.output_dir}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.patience:
                print("Early stopping.");
                break
    if is_main_process(): print("Training complete.")
    if is_ddp: cleanup_ddp()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description="Fine-tune Gemma-2B for causal language modeling")
    parser.add_argument("--model_name", type=str, default="model/gemma-2b")
    parser.add_argument("--train_file"     , type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--output_dir"     , type=str, default="model_finetuned/gemma_finetuned")
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input token length")
    parser.add_argument("--max_target_length",type=int, default=512, help="Max output token length")
    parser.add_argument("--batch_size"     , type=int, default=10)
    parser.add_argument("--epochs"         , type=int, default=10)
    parser.add_argument("--lr"             , type=float, default=5e-5)
    parser.add_argument("--weight_decay"   , type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm"  , type=float, default=1.0)
    parser.add_argument("--scheduler_type" , type=str, default="linear", choices=["linear","cosine"])
    parser.add_argument("--warmup_ratio"   , type=float, default=0.1)
    parser.add_argument("--patience"       , type=int, default=3)
    parser.add_argument("--use_torch_compile", action="store_true")
    parser.add_argument("--num_workers"    , type=int, default=4)
    parser.add_argument("--seed"           , type=int, default=42)
    args=parser.parse_args()
    main(args)
