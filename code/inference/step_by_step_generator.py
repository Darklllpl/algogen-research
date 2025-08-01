import json
from pathlib import Path
import copy
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FINETUNED_MODEL_PATH = "./model_finetuned/t5_base_delta_predictor_seed29" 

INPUT_SVL_DIR = "svl_dataset/mid_test"

OUTPUT_SVL_DIR = "llm_generated_output/t5-base_seed29"

MAX_FRAMES = 300

def load_model_and_tokenizer(model_path: str):
    """Load finetuned model and tokenizer."""
    print(f"Loading model and tokenizer from '{model_path}'...")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        print(f"Model loaded to: {device}")
        return model, tokenizer, device
    except OSError:
        print(f"Error: Model file not found at '{model_path}'.")
        return None, None, None

def format_input_for_t5(algorithm_info: dict, current_frame: dict, last_meta: dict) -> str:
    """Format current frame state into a string exactly matching finetuned training data."""
    algorithm_name = algorithm_info.get("name", "Unknown Algorithm")
    code_highlight_line_num = current_frame.get("code_highlight", 1)
    code_idx = code_highlight_line_num - 1
    pseudocode_list = current_frame.get("pseudocode", [])
    code_line_text = pseudocode_list[code_idx] if 0 <= code_idx < len(pseudocode_list) else ""
    data_state_obj = current_frame.get("data_state", {})
    
    if isinstance(data_state_obj, list):
        data_list = data_state_obj
    else:
        data_list = data_state_obj.get("data", [])
    
    compressed_data_state = {
        "v": [elem.get("value") for elem in data_list],
        "s": [elem.get("state") for elem in data_list]
    }
    
    context = {
        "meta": last_meta,
        "data_state": compressed_data_state,
        "boundaries": current_frame.get("boundaries", []),
        "temp_elements": current_frame.get("temp_elements", [])
    }
    context_str = json.dumps(context, separators=(',', ':'))

    prompt = (
        f"generate visual delta: "
        f"ALGORITHM: {algorithm_name} "
        f"CODE_LINE[{code_highlight_line_num}]: {code_line_text.strip()} "
        f"CONTEXT: {context_str}"
    )
    return prompt

def apply_delta_to_state(current_frame: dict, delta: dict, last_meta: dict) -> dict:
    """
    A more complete state update function that handles multiple state change operations.
    """
    new_frame = copy.deepcopy(current_frame)
    data_state = new_frame["data_state"]
    
    if isinstance(data_state, list):
        data = data_state
    else:
        data = data_state["data"]
    
    operations = delta.get("operations", [])
    for op_group in operations:
        op_list = op_group if isinstance(op_group, list) else [op_group]
        for op in op_list:
            if isinstance(op, dict):
                op_name, params = op.get("op"), op.get("params", {})
                
                if op_name == "updateStyle":
                    for i in params.get("indices", []):
                        if 0 <= i < len(data): data[i]["state"] = params.get("styleKey")
                
                elif op_name == "moveElements":
                    snapshot = copy.deepcopy(data)
                    for p in params.get("pairs", []):
                        src, dst = p.get("fromIndex"), p.get("toIndex")
                        if src is not None and dst is not None and 0 <= src < len(snapshot) and 0 <= dst < len(snapshot):
                            data[dst] = snapshot[src]

                elif op_name == "shiftElements":
                    snapshot = copy.deepcopy(data)
                    for s in params.get("shifts", []):
                        src, dst = s.get("fromIndex"), s.get("toIndex")
                        if src is not None and dst is not None and 0 <= src < len(snapshot) and 0 <= dst < len(snapshot):
                            data[dst] = snapshot[src]

                elif op_name == "updateValues":
                    for u in params.get("updates", []):
                        idx, val = u.get("index"), u.get("value")
                        if idx is not None and 0 <= idx < len(data):
                            data[idx]["value"] = val

                elif op_name == "drawTemp":
                    new_temp = {
                        "type": params.get("type"),
                        "indices": params.get("indices"),
                        "styleKey": params.get("styleKey")
                    }
                    new_frame["temp_elements"].append(new_temp)
                
                elif op_name == "removeTemp":
                    remove_type = params.get("type")
                    new_frame["temp_elements"] = [t for t in new_frame["temp_elements"] if t["type"] != remove_type]

    if "code_highlight" in delta:
        new_frame["code_highlight"] = delta.get("code_highlight")

    return new_frame

def decode_no_space(ids, tokenizer):
    """
    Remove '▁' from each token and concatenate into a string without spaces.
    """
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
    return ''.join(tok.lstrip('▁') for tok in tokens)

def is_delta_equal(delta1, delta2):
    return json.dumps(delta1, sort_keys=True) == json.dumps(delta2, sort_keys=True)

def main():
    model, tokenizer, device = load_model_and_tokenizer(FINETUNED_MODEL_PATH)
    if not model: return

    input_dir = Path(INPUT_SVL_DIR)
    output_dir = Path(OUTPUT_SVL_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    json_files = list(input_dir.glob('*.json'))
    if not json_files:
        print(f"Error: No JSON files found in '{input_dir}'."); return

    for input_path in json_files:
        print(f"\nProcessing file: {input_path}")

        output_path = output_dir / input_path.name
        if output_path.exists():
            print(f"File already exists, skipping: {output_path}")
            continue

        if not input_path.exists():
            print(f"Error: Input file not found -> {input_path}"); continue

        svl_data = json.loads(input_path.read_text(encoding='utf-8'))
        algorithm_info = svl_data["algorithm"]
        initial_frame = svl_data["initial_frame"]
        original_deltas = svl_data.get("deltas", [])
        
        current_frame = copy.deepcopy(initial_frame)
        current_frame["temp_elements"] = []
        current_frame["boundaries"] = []
        
        print("\nStarting autoregressive generation...")
        generated_deltas = []
        
        last_meta = {var['name']: "-" for var in initial_frame.get("variables_schema", [])}

        for frame_num in range(MAX_FRAMES):
            print(f"--- Generating Delta for frame {frame_num + 1} ---")
            
            input_text = format_input_for_t5(algorithm_info, current_frame, last_meta)
            print(f"Model input: {input_text}")
            
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids, max_length=512, eos_token_id=tokenizer.eos_token_id)
            delta_str = decode_no_space(outputs[0], tokenizer)
            
            if '}' in delta_str:
                delta_str = delta_str[:delta_str.rfind('}')+1]
            
            original_delta_str = delta_str
            if delta_str and not delta_str.startswith('{'):
                repaired_str = '{' + delta_str
                try:
                    json.loads(repaired_str)
                    delta_str = repaired_str
                except json.JSONDecodeError:
                    print(f"Failed to repair: Adding {{ still invalid, keeping original")
                    pass

            if not delta_str or delta_str.lower() == "<eos>":
                print("Model output sequence ended, stopping generation."); break
                
            print(f"Model output: {delta_str}")
            
            try:
                delta_obj = json.loads(delta_str)
                if frame_num < len(original_deltas):
                    if not is_delta_equal(delta_obj, original_deltas[frame_num]):
                        print(f"Frame {frame_num+1} delta does not match original delta, stopping processing for this file, but this frame will also be saved to json file for later analysis")
                        generated_deltas.append(delta_obj)
                        break
                generated_deltas.append(delta_obj)
                
                current_frame = apply_delta_to_state(current_frame, delta_obj, last_meta)
                last_meta.update(delta_obj.get("meta", {}))

            except json.JSONDecodeError as e:
                print(f"Error: Model output is not a valid JSON format. Error: {e}. Stopping generation."); break
                
        else:
            print(f"Warning: Maximum frame limit ({MAX_FRAMES} frames) reached, generation forced to stop.")

        final_svl_object = {
            "svl_version": "5.0", "algorithm": algorithm_info,
            "initial_frame": initial_frame, "deltas": generated_deltas
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_svl_object, f, indent=2, ensure_ascii=False)

        print(f"Generation complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()
