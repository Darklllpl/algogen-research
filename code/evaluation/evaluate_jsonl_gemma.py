import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import logging
import datetime

# --- 1. Helper Functions ---

def decode_no_space(ids, tokenizer):
    """
    Decode token IDs and remove all spaces.
    """
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    return tokenizer.decode(ids, skip_special_tokens=True).replace(' ', '')

# --- 2. Data Loader ---

class JSONLDataset(Dataset):
    """
    Custom dataset for loading .jsonl files.
    """
    def __init__(self, file_path):
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        """
        Load data from file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path.resolve()}")
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Parse a single sample ('text' field) into input/target.
        """
        item = self.data[idx]
        text = item.get('text', '')
        try:
            user_split = text.split('<start_of_turn>user\n', 1)
            if len(user_split) < 2:
                raise ValueError('No <start_of_turn>user marker')
            user_and_rest = user_split[1]
            user_input, rest = user_and_rest.split('<end_of_turn>\n<start_of_turn>model\n', 1)
            user_input = user_input.strip()
            model_output = rest.split('<eos>', 1)[0].strip()
            return user_input, model_output
        except Exception as e:
            raise ValueError(f"Failed to parse 'text' field at line {idx}: {e}\nRaw: {text}")

# --- 3. Evaluation Logic ---

def main(args):
    """
    Main evaluation function.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    base_output = Path(args.output_file)
    log_path = base_output.parent / f"{base_output.stem}_{timestamp}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Evaluation started: {timestamp}")
    logging.info(f"Args: {vars(args)}")
    gpu_list = [int(x) for x in args.gpus.split(',') if x.strip()] if args.gpus else None
    if gpu_list and torch.cuda.is_available():
        primary_gpu = gpu_list[0]
        device = torch.device(f"cuda:{primary_gpu}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading model and tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    if gpu_list and len(gpu_list) > 1:
        logging.info(f"Using GPUs {gpu_list} for evaluation")
        model = torch.nn.DataParallel(model, device_ids=gpu_list)
    elif torch.cuda.device_count() > 1 and not gpu_list:
        all_gpus = list(range(torch.cuda.device_count()))
        logging.info(f"Using all available GPUs {all_gpus} for evaluation")
        model = torch.nn.DataParallel(model)
    
    if isinstance(model, torch.nn.DataParallel):
        model.generate = model.module.generate
    model.eval()

    logging.info(f"Loading test set: {args.test_file}")
    dataset = JSONLDataset(args.test_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    em_correct = 0
    jse_correct = 0
    total_count = 0
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Evaluating on {len(dataset)} samples...")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs, targets = batch

                # Format prompt for Gemma
                prompts = [f"Input: {inp}\nOutput: " for inp in inputs]
                tokenized_inputs = tokenizer(
                    prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(device)

                generated_ids = model.generate(
                    **tokenized_inputs,
                    max_length=args.max_target_len,
                    num_beams=4,
                    early_stopping=True
                )
                
                predicted_texts = [decode_no_space(g, tokenizer) for g in generated_ids]
                
                for i in range(len(predicted_texts)):
                    pred_text = predicted_texts[i]
                    original_pred_text = pred_text
                    true_text = targets[i]
                    input_text = inputs[i]
                    
                    # If "model" separator appears, extract the content after it
                    separator = "model"
                    if separator in pred_text:
                        pred_text = pred_text.split(separator)[-1].strip()
                    
                    # Try to repair incomplete JSON if necessary
                    if pred_text and not pred_text.startswith('{') and pred_text.endswith('}'):
                        repaired_text = '{' + pred_text
                        try:
                            json.loads(repaired_text)
                            pred_text = repaired_text
                        except json.JSONDecodeError:
                            pass
                    
                    is_em_correct = (pred_text.strip() == true_text.strip())
                    if is_em_correct:
                        em_correct += 1

                    is_jse_correct = False
                    try:
                        pred_json = json.loads(pred_text)
                        true_json = json.loads(true_text)
                        if pred_json == true_json:
                            is_jse_correct = True
                            jse_correct += 1
                    except json.JSONDecodeError:
                        is_jse_correct = False
                    
                    result_line = {
                        "input": input_text,
                        "ground_truth_target": true_text,
                        "repaired_predicted_target": pred_text,
                        "is_em_correct": is_em_correct,
                        "is_jse_correct": is_jse_correct
                    }
                    outfile.write(json.dumps(result_line, ensure_ascii=False) + '\n')

                total_count += len(inputs)

    em_score = em_correct / total_count if total_count > 0 else 0
    jse_score = jse_correct / total_count if total_count > 0 else 0

    summary_text = f"""
--- Evaluation Summary ---
Total samples: {total_count}
Exact Match (EM): {em_score:.4f} ({em_correct}/{total_count})
JSON Semantic Equivalence (JSE): {jse_score:.4f} ({jse_correct}/{total_count})
-------------------------
"""
    logging.info(summary_text)
    print(summary_text)
    
    logging.info(f"Detailed evaluation log saved to: {output_path.resolve()}")
    summary_path = output_path.parent / f"{output_path.stem}_{timestamp}_summary.json"
    summary = {
        "timestamp": timestamp,
        "total_count": total_count,
        "em_correct": em_correct,
        "jse_correct": jse_correct,
        "em_score": em_score,
        "jse_score": jse_score,
        "args": vars(args)
    }
    with open(summary_path, 'w', encoding='utf-8') as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)
    logging.info(f"Evaluation summary saved to: {summary_path.resolve()}")
    
    txt_summary_path = output_path.parent / f"{output_path.stem}_{timestamp}_summary.txt"
    with open(txt_summary_path, 'w', encoding='utf-8') as tf:
        tf.write(summary_text)
    logging.info(f"Human-readable evaluation result saved to: {txt_summary_path.resolve()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Gemma model step prediction with .jsonl test set (with repair and separator handling)")
    parser.add_argument("--model_path", type=str, required=True, help="Directory of the fine-tuned model")
    parser.add_argument("--test_file", type=str, required=True, help="Test .jsonl file with 'text' field")
    parser.add_argument("--output_file", type=str, required=True, help="Output log file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--max_target_len", type=int, default=512, help="Max target length for generation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--gpus", type=str, default="0", help="GPU list, e.g., '0,1,2'; use all if not specified")
    args = parser.parse_args()
    main(args)
