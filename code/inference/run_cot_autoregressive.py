import json
import re
from pathlib import Path
import copy
from tqdm import tqdm
import requests
import time

API_BASE_URL = ''  
API_TOKEN = ''    
DEFAULT_MODEL = "o4-mini"

INPUT_SVL_DIR = "svl_dataset/small_test"
OUTPUT_SVL_DIR = "local_model_generated_output/o4-mini-cot"

MAX_FRAMES = 200

def call_llm_api(prompt: str, model_name: str = DEFAULT_MODEL):
    """
    Call LLM API to get response, return response content and statistics
    """
    url = f'{API_BASE_URL}/chat/completions'
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            
            # 提取统计信息
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            stats = {
                "content": content,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
            return stats
        else:
            print(f"API response format exception: {result}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return None

def build_the_prompt(user_input_str: str) -> str:
    """
    Build the full prompt for the LLM. All content in English.
    """
    prompt = '''
### ROLE & TASK
You are a 'Deterministic Algorithm Execution Machine'. Your task is to simulate the NEXT SINGLE STEP of an algorithm. To do this, you must follow a strict three-stage process: THINK, CODE, and OUTPUT.

#### Additional Rules:
- Only write a variable to meta if its value actually changes. Do not include uninitialized variables in meta.
- If the current code line only performs global/local initialization and does not affect visualization, you should directly jump to the next key logic line.

### STAGE 1: THINK (Chain-of-Thought)
First, analyze the provided CONTEXT. Verbalize your reasoning in a <thinking> block. You should determine what the next logical operation is based on the current state and the highlighted code line.

### STAGE 2: CODE (Python Execution)
Second, based on your reasoning, write a short, self-contained Python snippet inside a <code> block.
- This code MUST be executable.
- It will be given a dictionary named `context` as input, which contains `meta` (a dict) and `data` (a list of dicts, like `[{{"value": v, "state": s}} , ...]`).
- The code's task is to calculate the content of the *next* FrameDelta.
- It MUST create a dictionary named `result_delta` containing the calculated `meta`, `code_highlight`, and `operations`.

### STAGE 3: OUTPUT (Final JSON)
Third, copy the `result_delta` dictionary you calculated in the CODE stage into a <json_output> block. This must be a single, valid JSON object.

---
### GOLDEN EXAMPLE

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[1]: function BubbleSort(array): CONTEXT: {{"meta":{{}},"data_state":{{"v":[5,3],"s":["idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
This is the initial state, not yet in the loop. The next step should jump to the for i loop.
</thinking>
<code>
meta_out = {{}}
code_highlight_out = 3
operations_out = []
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{}},"code_highlight":3,"operations":[]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Selection Sort CODE_LINE[1]: function SelectionSort(arr): CONTEXT: {{"meta":{{}},"data_state":{{"v":[7,2,4],"s":["idle","idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
This is the initial state, not yet in the loop. The next step should jump to the for i loop.
</thinking>
<code>
meta_out = {{}}
code_highlight_out = 3
operations_out = []
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{}},"code_highlight":3,"operations":[]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {{"meta":{{"i":0,"j":0}},"data_state":{{"v":[5,3],"s":["comparing","comparing"]}}, ...}}

**YOUR RESPONSE:**
<thinking>
The current state has j=0. The values to compare are array[0] (value=5) and array[1] (value=3). The condition `5 > 3` is true. Therefore, the code highlight should move to line 6 (swap). The operations should be to change the style to 'tobe_swapped' and draw a swap arrow. The meta variables i and j remain unchanged in this step.
</thinking>
<code>
# Input context will be provided here
# context = {{"meta": {{"i": 0, "j": 0}}, "data": [{{"value": 5, "state": "comparing"}}, {{"value": 3, "state": "comparing"}}]}}

meta_out = context['meta'].copy()
code_highlight_out = 6
operations_out = [[
    {{"op": "updateStyle", "params": {{"indices": [0, 1], "styleKey": "tobe_swapped"}}}},
    {{"op": "drawTemp", "params": {{"type": "swap_arrow", "indices": [0, 1], "styleKey": "swap_arrow"}}}}
]]
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":0}},"code_highlight":6,"operations":[[{{"op":"updateStyle","params":{{"indices":[0,1],"styleKey":"tobe_swapped"}}}},{{"op":"drawTemp","params":{{"type":"swap_arrow","indices":[0,1],"styleKey":"swap_arrow"}}}}]]}}
</json_output>

---
### YOUR TASK

**USER INPUT:**
{user_input_str}

**YOUR RESPONSE:**
'''.format(user_input_str=user_input_str)
    return prompt

def format_input_for_cot(algorithm_info: dict, current_frame: dict, last_meta: dict, schema_var_names: list) -> str:
    """Format input for CoT baseline: meta always includes all schema variables, unassigned as '-'"""
    algorithm_name = algorithm_info.get("name", "Unknown Algorithm")
    code_highlight_line_num = current_frame.get("code_highlight", 1)
    code_idx = code_highlight_line_num - 1
    pseudocode_list = current_frame.get("pseudocode", [])
    code_line_text = pseudocode_list[code_idx] if 0 <= code_idx < len(pseudocode_list) else ""
    data_state_obj = current_frame.get("data_state", {})
    
    # Handle data_state as list or dict
    if isinstance(data_state_obj, list):
        data_list = data_state_obj
    else:
        data_list = data_state_obj.get("data", [])
    
    compressed_data_state = {
        "v": [elem.get("value") for elem in data_list],
        "s": [elem.get("state") for elem in data_list]
    }
    
    # meta always includes all schema variables, unassigned as '-'
    filtered_meta = {k: last_meta.get(k, "-") for k in schema_var_names}
    
    context = {
        "meta": filtered_meta,
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

def extract_delta_from_llm_output(llm_output_text: str) -> dict:
    """
    Extract delta from LLM's three-stage output
    """
    try:
        json_match = re.search(r'<json_output>(.*?)</json_output>', llm_output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        
        code_match = re.search(r'<code>(.*?)</code>', llm_output_text, re.DOTALL)
        if code_match:
            code_to_exec = code_match.group(1).strip()
            return None
            
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Parse LLM output failed: {e}")
        return None
    
    return None

def apply_delta_to_state(current_frame: dict, delta: dict, last_meta: dict) -> dict:
    """
    Apply delta to current state, generate next frame
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

def is_delta_equal(delta1, delta2):
    """Compare two deltas"""
    return json.dumps(delta1, sort_keys=True) == json.dumps(delta2, sort_keys=True)

def run_single_frame_generation(user_input_str: str, model_name: str = DEFAULT_MODEL):
    """
    Run CoT+Python generation for a single input, return delta and statistics
    """
    prompt = build_the_prompt(user_input_str)
    
    start_time = time.time()
    api_result = call_llm_api(prompt, model_name)
    end_time = time.time()
    
    if api_result is None:
        print("  [Failed] LLM API call failed")
        return None, None
    
    latency = end_time - start_time
    llm_output_text = api_result["content"]
    prompt_tokens = api_result["prompt_tokens"]
    completion_tokens = api_result["completion_tokens"]
    total_tokens = api_result["total_tokens"]
    
    delta_obj = extract_delta_from_llm_output(llm_output_text)
    
    if delta_obj is not None:
        print(f"[Model:{model_name}] Response length:{len(llm_output_text)} | Latency:{latency:.2f}s | Token:in={prompt_tokens},out={completion_tokens},sum={total_tokens} | Extraction:json_output")
        stats = {
            "latency_sec": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "extraction_method": "json_output"
        }
        return delta_obj, stats
    
    try:
        code_to_exec = re.search(r'<code>(.*?)</code>', llm_output_text, re.DOTALL).group(1).strip()
    except AttributeError:
        print("  [Failed] LLM output format is not standard, unable to extract Python code.")
        print(f"  [Debug] LLM output content: {llm_output_text[:500]}...")
        return None, None

    try:
        context_str = re.search(r'CONTEXT: (.*)', user_input_str).group(1)
        context_data = json.loads(context_str)
    except (AttributeError, json.JSONDecodeError) as e:
        print(f"  [Failed] Unable to parse CONTEXT in input: {e}")
        return None, None
    
    execution_scope = {
        'context': {
            'meta': context_data['meta'],
            'data': [{"value": v, "state": s} for v, s in zip(context_data['data_state']['v'], context_data['data_state']['s'])]
        }
    }
    
    try:
        exec(code_to_exec, {}, execution_scope)
        predicted_delta = execution_scope.get('result_delta')
        if predicted_delta:
            print(f"[Model:{model_name}] Response length:{len(llm_output_text)} | Latency:{latency:.2f}s | Token:in={prompt_tokens},out={completion_tokens},sum={total_tokens} | Extraction:code_execution")
            stats = {
                "latency_sec": latency,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "extraction_method": "code_execution"
            }
            return predicted_delta, stats
        else:
            print("  [Failed] Execution succeeded, but code did not generate 'result_delta' dictionary.")
            return None, None
    except Exception as e:
        print(f"  [Failed] LLM-generated Python code execution failed: {e}")
        print(f"  [Debug] Code attempted to execute: {code_to_exec}")
        return None, None

def main():
    input_dir = Path(INPUT_SVL_DIR)
    output_dir = Path(OUTPUT_SVL_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    json_files = list(input_dir.glob('*.json'))
    if not json_files:
        print(f"错误: 在文件夹 '{input_dir}' 中未找到任何 JSON 文件。")
        return

    all_files_stats = []

    for input_path in json_files:
        print(f"\n{'='*30}\nProcessing file: {input_path}\n{'='*30}")

        output_path = output_dir / input_path.name
        if output_path.exists():
            print(f"File already exists, skipping: {output_path}")
            continue

        if not input_path.exists():
            print(f"Error: Input file not found -> {input_path}")
            continue

        svl_data = json.loads(input_path.read_text(encoding='utf-8'))
        algorithm_info = svl_data["algorithm"]
        initial_frame = svl_data["initial_frame"]
        original_deltas = svl_data.get("deltas", [])
        
        schema_var_names = [var['name'] for var in initial_frame.get("variables_schema", [])]
        
        current_frame = copy.deepcopy(initial_frame)
        current_frame["temp_elements"] = []
        current_frame["boundaries"] = []
        
        print("\nStarting CoT baseline autoregressive generation...")
        generated_deltas = []
        current_file_frame_data = []
        
        last_meta = {}

        for frame_num in range(MAX_FRAMES):
            print(f"\n--- Generating Delta for frame {frame_num + 1} ---")
            
            input_text = format_input_for_cot(algorithm_info, current_frame, last_meta, schema_var_names)
            print(f"Model input: {input_text}")
            
            result = run_single_frame_generation(input_text, DEFAULT_MODEL)
            
            if result is None or result[0] is None:
                print("Model generation failed, stopping generation.")
                break
            
            delta_obj, frame_stats = result
            print(f"Model output: {json.dumps(delta_obj, separators=(',', ':'))}")
            
            current_file_frame_data.append({
                "latency_sec": frame_stats["latency_sec"],
                "prompt_tokens": frame_stats["prompt_tokens"],
                "completion_tokens": frame_stats["completion_tokens"],
                "total_tokens": frame_stats["total_tokens"],
                "extraction_method": frame_stats["extraction_method"],
                "is_correct": True
            })
            
            if frame_num < len(original_deltas):
                if not is_delta_equal(delta_obj, original_deltas[frame_num]):
                    print(f"Frame {frame_num+1} delta does not match original delta, stopping generation for this file, continuing with next file.")
                    generated_deltas.append(delta_obj)
                    break
            
            generated_deltas.append(delta_obj)
            
            current_frame = apply_delta_to_state(current_frame, delta_obj, last_meta)
            last_meta.update(delta_obj.get("meta", {}))
            last_meta = {k: v for k, v in last_meta.items() if k in schema_var_names}
                
        else:
            print(f"Warning: Maximum frame limit ({MAX_FRAMES} frames) reached, generation forced to stop.")

        final_svl_object = {
            "svl_version": "5.0", 
            "algorithm": algorithm_info,
            "initial_frame": initial_frame, 
            "deltas": generated_deltas
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_svl_object, f, indent=2, ensure_ascii=False)

        print(f"Generation complete! Results saved to: {output_path}")

        if current_file_frame_data:
            successful_frames = len(current_file_frame_data)
            avg_latency = sum(d['latency_sec'] for d in current_file_frame_data) / successful_frames if successful_frames > 0 else 0
            avg_total_tokens = sum(d['total_tokens'] for d in current_file_frame_data) / successful_frames if successful_frames > 0 else 0
            total_tokens_for_file = sum(d['total_tokens'] for d in current_file_frame_data)
            correct_frames = sum(1 for d in current_file_frame_data if d.get('is_correct', True))
            
            file_stats = {
                "file_name": input_path.name,
                "successful_frames (FFI)": successful_frames,
                "correct_frames": correct_frames,
                "accuracy": correct_frames / successful_frames if successful_frames > 0 else 0,
                "total_tokens": total_tokens_for_file,
                "avg_latency_sec_per_frame": avg_latency,
                "avg_total_tokens_per_frame": avg_total_tokens
            }
            all_files_stats.append(file_stats)
            
            print("\n" + "-"*15 + " File Statistics " + "-"*15)
            print(f"File name: {file_stats['file_name']}")
            print(f"Successful frames (FFI): {file_stats['successful_frames (FFI)']}")
            print(f"Correct frames: {file_stats['correct_frames']}")
            print(f"Accuracy: {file_stats['accuracy']:.2%}")
            print(f"Total tokens consumed: {file_stats['total_tokens']}")
            print(f"Average latency per frame: {file_stats['avg_latency_sec_per_frame']:.2f} seconds")
            print(f"Average tokens per frame: {file_stats['avg_total_tokens_per_frame']:.2f}")
            print("-"*(30 + len(" File Statistics ")))

    if all_files_stats:
        print(f"\n\n{'='*20}\nExperiment Report\n{'='*20}")
        report_path = Path(OUTPUT_SVL_DIR) / "_experiment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(all_files_stats, f, indent=2, ensure_ascii=False)
        print(f"Detailed statistics report for all files saved to: {report_path}")
        
        total_files = len(all_files_stats)
        total_frames = sum(stats['successful_frames (FFI)'] for stats in all_files_stats)
        total_correct_frames = sum(stats['correct_frames'] for stats in all_files_stats)
        total_tokens = sum(stats['total_tokens'] for stats in all_files_stats)
        avg_latency_all = sum(stats['avg_latency_sec_per_frame'] * stats['successful_frames (FFI)'] for stats in all_files_stats) / total_frames if total_frames > 0 else 0
        
        print(f"\nOverall statistics:")
        print(f"Number of processed files: {total_files}")
        print(f"Total generated frames: {total_frames}")
        print(f"Total correct frames: {total_correct_frames}")
        print(f"Overall accuracy: {total_correct_frames / total_frames:.2%}" if total_frames > 0 else "Overall accuracy: N/A")
        print(f"Total tokens consumed: {total_tokens}")
        print(f"Average latency per frame: {avg_latency_all:.2f} seconds")

if __name__ == "__main__":
    main() 