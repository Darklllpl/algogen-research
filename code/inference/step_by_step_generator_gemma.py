import json
from pathlib import Path
import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FINETUNED_MODEL_PATH = "./model_finetuned/gemma_finetuned_seed13"  
INPUT_SVL_JSON_DIR = "svl_dataset/all_test"
OUTPUT_SVL_JSON_DIR = "llm_generated_output/all/gemma_seed13"

MAX_FRAMES = 300 

def load_model_and_tokenizer(model_path: str):
    """Load finetuned Gemma model and tokenizer."""
    print(f"Loading model and tokenizer from '{model_path}'...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model loaded to: {device}")
        return model, tokenizer, device
    except OSError:
        print(f"Error: Model file not found at '{model_path}'.")
        return None, None, None

def format_input_for_gemma(algorithm_info: dict, current_frame: dict, last_meta: dict) -> str:
    """
    Format current frame state into a conversation format exactly matching Gemma finetuned data.
    """
    algorithm_name = algorithm_info.get("name", "Unknown Algorithm")
    code_highlight_line_num = current_frame.get("code_highlight", 1)
    code_idx = code_highlight_line_num - 1
    pseudocode_list = current_frame.get("pseudocode", [])
    code_line_text = pseudocode_list[code_idx] if 0 <= code_idx < len(pseudocode_list) else ""
    
    data_state_obj = current_frame.get("data_state", [])
    
    compressed_data_state = {
        "v": [elem.get("value") for elem in data_state_obj],
        "s": [elem.get("state") for elem in data_state_obj]
    }
    
    context = {
        "meta": last_meta, "data_state": compressed_data_state,
        "boundaries": [], "temp_elements": []
    }
    context_str = json.dumps(context, separators=(',', ':'))

    user_prompt = (
        f"generate visual delta: ALGORITHM: {algorithm_name} "
        f"CODE_LINE[{code_highlight_line_num}]: {code_line_text.strip()} "
        f"CONTEXT: {context_str}"
    )
    
    final_prompt = (
        f"<start_of_turn>user\n"
        f"{user_prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    return final_prompt

def apply_delta_to_state(current_frame: dict, delta: dict, last_meta: dict) -> dict:
    """
    A more complete state update function that handles multiple state change operations.
    """
    new_frame = copy.deepcopy(current_frame)
    if "temp_elements" not in new_frame:
        new_frame["temp_elements"] = []
    if "boundaries" not in new_frame:
        new_frame["boundaries"] = []
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

def is_delta_equal(delta1, delta2):
    return json.dumps(delta1, sort_keys=True) == json.dumps(delta2, sort_keys=True)

def main():
    model, tokenizer, device = load_model_and_tokenizer(FINETUNED_MODEL_PATH)
    if not model: return

    input_dir = Path(INPUT_SVL_JSON_DIR)
    output_dir = Path(OUTPUT_SVL_JSON_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    if not input_dir.exists():
        print(f"Error: Input directory not found -> {input_dir}"); return

    for input_path in sorted(input_dir.glob("*.json")):
        print(f"\nProcessing file: {input_path.name}")
        output_path = output_dir / input_path.name
        if output_path.exists():
            print(f"Output file already exists, skipping: {output_path}")
            continue
        svl_data = json.loads(input_path.read_text(encoding='utf-8'))
        algorithm_info = svl_data["algorithm"]
        initial_frame = svl_data["initial_frame"]
        original_deltas = svl_data.get("deltas", [])
        max_frames = len(original_deltas)
        correct_count = 0
        ffi = None

        print("Starting autoregressive generation...")
        generated_deltas = []
        current_frame = copy.deepcopy(initial_frame)
        if "temp_elements" not in current_frame:
            current_frame["temp_elements"] = []
        if "boundaries" not in current_frame:
            current_frame["boundaries"] = []
        last_meta = {var['name']: "-" for var in initial_frame.get("variables_schema", [])}

        for frame_num in range(max_frames):
            print(f"--- Generating Delta for frame {frame_num + 1} ---")
            
            input_text = format_input_for_gemma(algorithm_info, current_frame, last_meta)
            print(f"Model input:\n{input_text}")
            
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(input_ids, max_length=len(input_ids[0]) + 512, eos_token_id=tokenizer.eos_token_id)
            generated_ids = outputs[0][len(input_ids[0]):]
            delta_str = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print(f"Model output: {delta_str}")
            
            try:
                delta_obj = json.loads(delta_str)
                if frame_num < len(original_deltas):
                    if is_delta_equal(delta_obj, original_deltas[frame_num]):
                        generated_deltas.append(delta_obj)
                        correct_count += 1
                    else:
                        ffi = frame_num + 1
                        print(f"Frame {ffi} delta does not match original delta, but continue processing, this frame will also be saved to json file for later analysis")
                        generated_deltas.append(delta_obj)
                
                is_final_step = any(op.get("params", {}).get("styleKey") == "sorted" and len(op.get("params", {}).get("indices", [])) == len(current_frame["data_state"]) for group in delta_obj.get("operations", []) for op in (group if isinstance(group, list) else [group]) if isinstance(op, dict))
                if is_final_step:
                    print("Detected algorithm completion operation, stopping generation.")
                    break

                current_frame = apply_delta_to_state(current_frame, delta_obj, last_meta)
                last_meta.update(delta_obj.get("meta", {}))

            except json.JSONDecodeError:
                print(f"Error: Model output is not a valid JSON format. Stopping generation."); break
                
        else:
            print(f"Warning: Maximum frame limit ({max_frames} frames) reached, generation forced to stop.")

        final_svl_object = {
            "svl_version": "4.0",
            "algorithm": algorithm_info,
            "initial_frame": initial_frame,
            "deltas": generated_deltas
        }
        output_path = output_dir / input_path.name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_svl_object, f, indent=2, ensure_ascii=False)

        print(f"\nGeneration complete! Results saved to: {output_path}")
        stats = {
            "file": input_path.name,
            "num_original_frames": len(original_deltas),
            "correct_frames": correct_count,
            "ffi": ffi,
            "correct_ratio": correct_count / len(original_deltas) if original_deltas else None
        }
        stats_path = output_dir / f"{input_path.stem}_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to: {stats_path}")

if __name__ == "__main__":
    main()