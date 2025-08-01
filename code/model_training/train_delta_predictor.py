import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_scheduler
import json
import argparse
from pathlib import Path
from tqdm.auto import tqdm
import random
import numpy as np
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Seq2SeqJSONLDataset(Dataset):
    """
    A generic class for processing JSON Lines format sequence-to-sequence datasets.
    """
    def __init__(self, file_path, tokenizer, max_input_len=256, max_target_len=256):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path.resolve()}")
        with path.open("r", encoding="utf-8") as f:
            return [
                json.loads(line)
                for line in f
                if line.strip() and 'input' in json.loads(line) and 'target' in json.loads(line)
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = str(item['input'])
        target_text = str(item['target'])

        input_enc = self.tokenizer(
            input_text,
            padding='max_length',
            max_length=self.max_input_len,
            truncation=True,
            return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text,
            padding='max_length',
            max_length=self.max_target_len,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input_enc.input_ids.squeeze()
        attention_mask = input_enc.attention_mask.squeeze()
        labels = target_enc.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def decode_no_space(ids, tokenizer):
    """
    Use tokenizer.decode directly to decode and try to parse as JSON, preserving the first '{';
    If parsing fails, remove the SentencePiece prefix '▁' and concatenate the remaining tokens.
    """
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    try:
        obj = json.loads(text)
        return json.dumps(obj, ensure_ascii=False, separators=(',',':'))
    except json.JSONDecodeError:
        tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        return ''.join(tok.lstrip('▁') for tok in tokens)

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, args, epoch, tokenizer):
    model.train()
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    progress_bar = tqdm(dataloader,
                        desc=f"Training (Epoch {epoch+1})",
                        leave=False,
                        disable=not is_main_process())

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.autocast(device_type=device.type,
                            dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
                            enabled=(device.type=='cuda')):
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps

        if is_main_process() and (step + 1) % 100 == 0:
            module = model.module if isinstance(model, DDP) else model
            decoded_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
            generated_ids = module.generate(
                input_ids=input_ids[:1],
                attention_mask=attention_mask[:1],
                max_length=args.max_target_len,
                num_beams=1,
                do_sample=False
            )
            decoded_pred = decode_no_space(generated_ids[0], tokenizer)
            print(f"\n--- Epoch{epoch+1} Step{step+1} Example Prediction ---")
            print("Input:", decoded_input)
            print("Predicted Output:", decoded_pred)
            print("-" * 40)

        if is_main_process():
            progress_bar.set_postfix({
                "loss": total_loss / (step + 1),
                "lr": scheduler.get_last_lr()[0]
            })

    return total_loss / len(dataloader)

def evaluate(model, dataloader, tokenizer, device, args):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    val_bar = tqdm(dataloader,
                   desc="Validation",
                   leave=False,
                   disable=not is_main_process())
    with torch.no_grad():
        for batch in val_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
                                enabled=(device.type=='cuda')):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                total_loss += outputs.loss.item()
            batches += 1

            module = model.module if isinstance(model, DDP) else model
            gen_ids = module.generate(input_ids, attention_mask=attention_mask,
                                      max_length=args.max_target_len)
            preds = [decode_no_space(ids, tokenizer) for ids in gen_ids]
            labels[labels == -100] = tokenizer.pad_token_id
            golds = [decode_no_space(ids, tokenizer) for ids in labels]
            for p, g in zip(preds, golds):
                if p.strip() == g.strip():
                    correct += 1
            total += len(golds)

            if is_main_process():
                val_bar.set_postfix({"loss": total_loss / batches})

    if dist.is_initialized():
        stats = torch.tensor([total_loss, batches, correct, total]).to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_loss, batches, correct, total = stats.tolist()

    avg_loss = total_loss / batches if batches > 0 else 0
    em_score = correct / total if total > 0 else 0
    return avg_loss, em_score

def main(args):
    is_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if is_ddp:
        setup_ddp()
    set_seed(args.seed)

    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}") if is_ddp else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_main_process():
        print(f"Distributed: {is_ddp}, Device: {device}")
        print(f"Loading model/tokenizer: {args.model_name}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, legacy=False, local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, local_files_only=True)

    if is_main_process():
        print("Adding '{' and '}' as normal tokens...")
    added = tokenizer.add_tokens(['{', '}'])
    if is_main_process():
        print(f"Added {added} tokens.")
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    if is_ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])], find_unused_parameters=False)

    if args.use_torch_compile and hasattr(torch, 'compile'):
        if is_main_process():
            print("Compiling model with torch.compile()")
        model = torch.compile(model)

    if is_main_process():
        print("Loading datasets...")
    train_ds = Seq2SeqJSONLDataset(args.train_file, tokenizer, args.max_input_len, args.max_target_len)
    val_ds   = Seq2SeqJSONLDataset(args.validation_file, tokenizer, args.max_input_len, args.max_target_len)
    train_sampler = DistributedSampler(train_ds) if is_ddp else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    train_dl = DataLoader(train_ds,
                          batch_size=args.batch_size,
                          sampler=train_sampler,
                          shuffle=(train_sampler is None),
                          num_workers=args.num_workers,
                          pin_memory=True)
    val_dl   = DataLoader(val_ds,
                          batch_size=args.batch_size * 2,
                          sampler=val_sampler,
                          num_workers=args.num_workers,
                          pin_memory=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * (len(train_dl) // args.gradient_accumulation_steps)
    scheduler = get_scheduler(
        name=args.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    best_loss = float('inf')
    epochs_no_improve = 0
    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print("Starting training...")

    for epoch in range(args.epochs):
        if is_main_process():
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_dl, optimizer, scheduler, scaler, device, args, epoch, tokenizer)
        if is_ddp:
            dist.barrier()
        if is_main_process():
            print(f"Epoch {epoch+1} train loss: {train_loss:.4f}")
        val_loss, em = evaluate(model, val_dl, tokenizer, device, args)
        if is_main_process():
            print(f"Val loss: {val_loss:.4f} | EM: {em:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                unwrapped = model.module if isinstance(model, DDP) else model
                if hasattr(unwrapped, '_orig_mod'):
                    unwrapped = unwrapped._orig_mod
                unwrapped.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                with open(Path(args.output_dir)/"training_args.json", 'w') as f:
                    json.dump(vars(args), f, indent=2)
                print(f"Saved new best model (loss={best_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs.")
            if epochs_no_improve >= args.patience:
                print("Early stopping.")
                break

    if is_main_process():
        print("Training complete.")
    if is_ddp:
        cleanup_ddp()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune T5 for Code+Context→DeltaJSON")
    parser.add_argument("--model_name", type=str, default="t5-base")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--validation_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="model_finetuned/t5_base_delta_predictor")
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--max_target_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["linear","cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--use_torch_compile", action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    main(args)
