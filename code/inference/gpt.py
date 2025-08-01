import json
from pathlib import Path
import copy
import os
import time
import requests 


API_ENDPOINT_URL = "YOUR_API_ENDPOINT_URL"
API_TOKEN = "YOUR_API_TOKEN"

MODEL_NAME = "gpt-4.1-2025-04-14"

INPUT_SVL_DIR = "svl_dataset/small_test"

OUTPUT_SVL_DIR = "local_model_generated_output/gpt-4.1-2025-04-14"

MAX_FRAMES = 10


def call_local_api(url: str, token: str, model_name: str, messages: list):
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.1,
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=180.0)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling local API: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error: API returned invalid JSON format. Received content: {response.text}")
        return None

def build_prompt_messages():

    system_prompt = """You are a 'Deterministic Algorithm Execution Machine'. Your ONLY purpose is to simulate algorithm execution step-by-step with 100% accuracy based on formal rules. You must adhere to the following rules STRICTLY:

RULE 1: OUTPUT FORMAT: Your entire output MUST be a single, raw, valid JSON object. It must start with '{' and end with '}'. NEVER wrap your output in ```json ... ```. NEVER add any explanatory text, acknowledgements, or apologies.

RULE 2: LOGICAL FLOW & STATE: You must track the program's execution flow.
    - An instruction's effects must be reflected in the next state.
    - After an instruction is executed, the `code_highlight` MUST ADVANCE to the next logical line. Do not get stuck on a line whose action is complete.
    - Loop variables (like 'i', 'j') MUST be updated accurately. For a `for j` loop, `j` is incremented ONLY after all operations for the current `j` (including comparison and any resulting swap) are complete. The `if` check and the `swap` action themselves DO NOT increment `j`.

RULE 3: SYNTAX: You must use the EXACT operation names ('op') and parameter keys ('params') as demonstrated in the examples. Do not invent new operations like 'drawBoundary'. Use 'updateBoundary' for creating or modifying boundaries.

RULE 4: OPERATION STRUCTURE: For actions that are a single logical step involving multiple elements (e.g., comparing two elements, initializing a partition), group the operations in a nested array: "operations": [[{op1}, {op2}]] . For sequential cleanup or state updates after a primary action, you can use a flat array: "operations": [{op1}, {op2}]. Follow the golden examples precisely.
"""
    
    # "黄金"示例集：13个经过精心挑选、逻辑无懈可击的范例
    golden_few_shot_examples = [
        # 1. 算法开始
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[1]: function BubbleSort(array): CONTEXT: {\"meta\":{\"i\":\"-\",\"j\":\"-\"},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{},\"code_highlight\":3,\"operations\":[]}"},

        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[3]: for i from 0 to n-1: CONTEXT: {\"meta\":{\"i\":\"-\",\"j\":\"-\"},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":\"-\"},\"code_highlight\":4,\"operations\":[]}"},

        # 2. Bubble Sort - 内循环初始化与并行比较
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[4]: for j from 0 to n-i-2: CONTEXT: {\"meta\":{\"i\":0,\"j\":\"-\"},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":0},\"code_highlight\":5,\"operations\":[[{\"op\":\"updateStyle\",\"params\":{\"indices\":[0],\"styleKey\":\"comparing\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[1],\"styleKey\":\"comparing\"}}]]}"},

        # 3. Bubble Sort - 决定交换 (if判断为真), j保持不变
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {\"meta\":{\"i\":0,\"j\":0},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"comparing\",\"comparing\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":0},\"code_highlight\":6,\"operations\":[[{\"op\":\"updateStyle\",\"params\":{\"indices\":[0,1],\"styleKey\":\"tobe_swapped\"}},{\"op\":\"drawTemp\",\"params\":{\"type\":\"swap_arrow\",\"indices\":[0,1],\"styleKey\":\"swap_arrow\"}}]]}"},
        
        # 4. Bubble Sort - 执行交换, j保持不变
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[6]: swap(array[j], array[j+1]) CONTEXT: {\"meta\":{\"i\":0,\"j\":0},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"tobe_swapped\",\"tobe_swapped\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[{\"type\":\"swap_arrow\",\"indices\":[0,1],\"styleKey\":\"swap_arrow\"}]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":0},\"code_highlight\":6,\"operations\":[{\"op\":\"removeTemp\",\"params\":{\"type\":\"swap_arrow\"}},{\"op\":\"moveElements\",\"params\":{\"pairs\":[{\"fromIndex\":0,\"toIndex\":1},{\"fromIndex\":1,\"toIndex\":0}],\"animationKey\":\"swap_animation\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[0,1],\"styleKey\":\"idle\"}}]}"},

        # 5. Bubble Sort - 交换完成后, j才递增
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[6]: swap(array[j], array[j+1]) CONTEXT: {\"meta\":{\"i\":0,\"j\":0},\"data_state\":{\"v\":[3,5,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":1},\"code_highlight\":5,\"operations\":[[{\"op\":\"updateStyle\",\"params\":{\"indices\":[1],\"styleKey\":\"comparing\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[2],\"styleKey\":\"comparing\"}}]]}"},

        # 6. Bubble Sort - 决定不交换 (if判断为假), j保持不变
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {\"meta\":{\"i\":0,\"j\":1},\"data_state\":{\"v\":[3,5,8,4,2],\"s\":[\"idle\",\"comparing\",\"comparing\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":1},\"code_highlight\":5,\"operations\":[{\"op\":\"updateStyle\",\"params\":{\"indices\":[1,2],\"styleKey\":\"idle\"}}]}"},

        # 7. Bubble Sort - 不交换完成后, j才递增
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {\"meta\":{\"i\":0,\"j\":1},\"data_state\":{\"v\":[3,5,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":0,\"j\":2},\"code_highlight\":5,\"operations\":[[{\"op\":\"updateStyle\",\"params\":{\"indices\":[2],\"styleKey\":\"comparing\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[3],\"styleKey\":\"comparing\"}}]]}"},

        # 8. Insertion Sort - 提取key (drawTemp)
        {"role": "user", "content": "generate visual delta: ALGORITHM: Insertion Sort CODE_LINE[2]: for i from 1 to n-1: CONTEXT: {\"meta\":{\"i\":\"-\",\"j\":\"-\",\"key\":\"-\"},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":1,\"key\":3,\"j\":\"-\"},\"code_highlight\":3,\"operations\":[[{\"op\":\"drawTemp\",\"params\":{\"type\":\"key_holder\",\"value\":3,\"styleKey\":\"key_holder\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[1],\"styleKey\":\"placeholder\"}}]]}"},
        
        # 9. Insertion Sort - 平移 (shiftElements)
        {"role": "user", "content": "generate visual delta: ALGORITHM: Insertion Sort CODE_LINE[5]: while j >= 0 and key < array[j]: CONTEXT: {\"meta\":{\"i\":1,\"j\":0,\"key\":3},\"data_state\":{\"v\":[5,3,8,4,2],\"s\":[\"comparing\",\"placeholder\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":1,\"key\":3,\"j\":0},\"code_highlight\":6,\"operations\":[[{\"op\":\"shiftElements\",\"params\":{\"shifts\":[{\"fromIndex\":0,\"toIndex\":1,\"value\":5}],\"animationKey\":\"shift_animation\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[0],\"styleKey\":\"placeholder\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[1],\"styleKey\":\"shifting\"}}]]}"},

        # 10. Insertion Sort - 插入key (updateValues)
        {"role": "user", "content": "generate visual delta: ALGORITHM: Insertion Sort CODE_LINE[5]: while j >= 0 and key < array[j]: CONTEXT: {\"meta\":{\"i\":1,\"j\":-1,\"key\":3},\"data_state\":{\"v\":[5,5,8,4,2],\"s\":[\"placeholder\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"i\":1,\"key\":3,\"j\":-1},\"code_highlight\":8,\"operations\":[[{\"op\":\"removeTemp\",\"params\":{\"type\":\"key_holder\"}},{\"op\":\"updateValues\",\"params\":{\"updates\":[{\"index\":0,\"value\":3}]}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[0],\"styleKey\":\"idle\"}}]]}"},

        # 11. Quick Sort - 初始分区 (updateBoundary)
        {"role": "user", "content": "generate visual delta: ALGORITHM: Quick Sort (Lomuto Partition) CODE_LINE[1]: function quickSort(array, low, high): CONTEXT: {\"meta\":{\"low\":\"-\",\"high\":\"-\",\"pivot\":\"-\",\"i\":\"-\",\"j\":\"-\"},\"data_state\":{\"v\":[5,6,1,9,7],\"s\":[\"idle\",\"idle\",\"idle\",\"idle\",\"idle\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"low\":0,\"high\":4,\"pivot\":\"-\",\"i\":\"-\",\"j\":\"-\"},\"code_highlight\":2,\"operations\":[[{\"op\":\"updateBoundary\",\"params\":{\"type\":\"partition_box\",\"range\":[0,4],\"label\":\"Partitioning\",\"styleKey\":\"partition_box\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[0,1,2,3,4],\"styleKey\":\"sub_array_active\"}}]]}"},
        
        # 12. Quick Sort - 结束分区 (removeBoundary)
        {"role": "user", "content": "generate visual delta: ALGORITHM: Quick Sort (Lomuto Partition) CODE_LINE[14]: swap(array[i+1], array[high]) CONTEXT: {\"meta\":{\"low\":0,\"high\":4,\"pivot\":7,\"i\":2,\"j\":\"-\"},\"data_state\":{\"v\":[5,6,1,7,9],\"s\":[\"partition_boundary\",\"partition_boundary\",\"sub_array_active\",\"tobe_swapped\",\"tobe_swapped\"]},\"boundaries\":[{\"type\":\"partition_box\",\"range\":[0,4],\"label\":\"Partitioning\",\"styleKey\":\"partition_box\"}],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{\"low\":0,\"high\":4,\"pivot\":7,\"i\":2,\"j\":\"-\"},\"code_highlight\":15,\"operations\":[{\"op\":\"removeBoundary\",\"params\":{\"type\":\"partition_box\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[0,1,2,3,4],\"styleKey\":\"idle\"}},{\"op\":\"updateStyle\",\"params\":{\"indices\":[3],\"styleKey\":\"sorted\"}}]}"},
        
        # 13. 算法终止
        {"role": "user", "content": "generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[3]: for i from 0 to n-1: CONTEXT: {\"meta\":{\"i\":4,\"j\":\"-\"},\"data_state\":{\"v\":[2,3,4,5,8],\"s\":[\"sorted\",\"sorted\",\"sorted\",\"sorted\",\"sorted\"]},\"boundaries\":[],\"temp_elements\":[]}"},
        {"role": "assistant", "content": "{\"meta\":{},\"code_highlight\":7,\"operations\":[]}"}
    ]
    
    messages = [{"role": "system", "content": system_prompt}] + golden_few_shot_examples
    return messages

initial_frame = {}

def format_input_for_prompt(algorithm_info: dict, current_frame: dict, last_meta: dict) -> str:
    algorithm_name = algorithm_info.get("name", "Unknown Algorithm")
    code_highlight_line_num = current_frame.get("code_highlight", 1)
    
    pseudocode_list = initial_frame.get("pseudocode", [])
    code_line_text = ""
    if pseudocode_list and 0 <= code_highlight_line_num - 1 < len(pseudocode_list):
        code_line_text = pseudocode_list[code_highlight_line_num - 1]
    
    data_state_obj = current_frame.get("data_state", {})
    
    data_list = []
    if isinstance(data_state_obj, list):
        data_list = data_state_obj
    else:
        if "data" in data_state_obj:
             data_list = data_state_obj["data"]
        elif "v" in data_state_obj and "s" in data_state_obj:
             data_list = [{"value": v, "state": s, "index": i} for i, (v, s) in enumerate(zip(data_state_obj["v"], data_state_obj["s"]))]

    compressed_data_state = {"v": [e.get("value") for e in data_list], "s": [e.get("state") for e in data_list]}
    context = {"meta": last_meta, "data_state": compressed_data_state, "boundaries": current_frame.get("boundaries", []), "temp_elements": current_frame.get("temp_elements", [])}
    context_str = json.dumps(context, separators=(',', ':'))
    return (
        f"generate visual delta: "
        f"ALGORITHM: {algorithm_name} "
        f"CODE_LINE[{code_highlight_line_num}]: {code_line_text.strip()} "
        f"CONTEXT: {context_str}"
    )

def apply_delta_to_state(current_frame: dict, delta: dict) -> dict:
    state = copy.deepcopy(current_frame)
    
    meta = delta.get('meta', {})
    if 'meta' not in state:
        state['meta'] = {}
    state['meta'].update(meta)
    
    state['code_highlight'] = delta.get('code_highlight', state.get('code_highlight'))
    
    data_list = state.get('data_state', [])

    for op_group in delta.get('operations', []):
        ops = op_group if isinstance(op_group, list) else [op_group]
        for op in ops:
            if not isinstance(op, dict): continue
            
            name = op.get('op')
            params = op.get('params', {})
            if not params: continue

            if name == 'updateStyle':
                idxs, style = params.get('indices', []), params.get('styleKey')
                if style:
                    for i in idxs:
                        if 0 <= i < len(data_list):
                            data_list[i]['state'] = style
            
            elif name == 'moveElements':
                snap = [e.copy() for e in data_list]
                for p in params.get('pairs', []):
                    fr, to = p.get('fromIndex'), p.get('toIndex')
                    if fr is not None and to is not None and 0 <= fr < len(snap) and 0 <= to < len(data_list):
                        moved_element = snap[fr]
                        data_list[to] = {
                            'index': to,
                            'value': moved_element['value'],
                            'state': moved_element['state']
                        }

            elif name == 'shiftElements':
                snap = [e.copy() for e in data_list]
                for s in params.get('shifts', []):
                    fr, to = s.get('fromIndex'), s.get('toIndex')
                    if fr is not None and to is not None and 0 <= fr < len(snap) and 0 <= to < len(data_list):
                        data_list[to] = {
                            'index': to,
                            'value': snap[fr]['value'],
                            'state': snap[fr]['state']
                        }

            elif name == 'updateValues':
                updates = params.get('updates', [])
                snap = [e.copy() for e in data_list]
                for u in updates:
                    idx, val = u.get('index'), u.get('value')
                    if idx is not None and val is not None and 0 <= idx < len(data_list):
                        data_list[idx] = {
                            'index': idx,
                            'value': val,
                            'state': snap[idx]['state']
                        }
            
            elif name == 'updateBoundary':
                t = params.get('type')
                if 'boundaries' not in state: state['boundaries'] = []
                state['boundaries'] = [b for b in state['boundaries'] if b.get('type') != t]
                state['boundaries'].append(params)

            elif name == 'removeBoundary':
                t = params.get('type')
                if 'boundaries' in state:
                    state['boundaries'] = [b for b in state['boundaries'] if b.get('type') != t]

            elif name == 'drawTemp':
                if 'temp_elements' not in state: state['temp_elements'] = []
                state['temp_elements'].append(params)

            elif name == 'removeTemp':
                t = params.get('type')
                if 'temp_elements' in state:
                    state['temp_elements'] = [e for e in state['temp_elements'] if e.get('type') != t]

    for i, el in enumerate(data_list):
        el['index'] = i

    return state

def is_termination_delta(delta: dict, final_line_num: int) -> bool:
    ops = delta.get("operations", [])
    highlight = delta.get("code_highlight")
    return not ops and highlight is not None and highlight >= final_line_num

def main():
    # 其余涉及API_TOKEN、API_ENDPOINT_URL的地方均已用占位符替换
    if "YOUR_API_KEY" in API_TOKEN or "sk-" != API_TOKEN:
         print("Error: API Token is invalid or using example token. Please set your real API_TOKEN in the code.")
         return

    base_messages = build_prompt_messages()
    input_dir = Path(INPUT_SVL_DIR)
    output_dir = Path(OUTPUT_SVL_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    json_files = sorted(list(input_dir.glob('*.json')))

    if not json_files:
        print(f"Error: No JSON files found in folder '{input_dir}'."); return
    
    all_files_stats = []

    for input_path in json_files:
        print(f"\n{'='*30}\nProcessing file: {input_path}\n{'='*30}")

        output_path = output_dir / input_path.name
        if output_path.exists():
            print(f"File already exists, skipping: {output_path}"); continue

        global initial_frame
        svl_data = json.loads(input_path.read_text(encoding='utf-8'))
        algorithm_info = svl_data["algorithm"]
        initial_frame = svl_data["initial_frame"]
        
        final_line_num = len(initial_frame.get("pseudocode", []))
        
        current_frame = copy.deepcopy(initial_frame)
        print("\nStarting self-regenerating...")
        generated_deltas = []
        last_meta = {var['name']: "-" for var in initial_frame.get("variables_schema", [])}
        current_file_frame_data = []
        
        for frame_num in range(MAX_FRAMES):
            print(f"\n--- Generating Delta for frame {frame_num + 1} ---")
            
            input_str = format_input_for_prompt(algorithm_info, current_frame, last_meta)
            messages_for_this_turn = base_messages + [{"role": "user", "content": input_str}]

            try:
                start_time = time.time()
                completion = call_local_api(API_ENDPOINT_URL, API_TOKEN, MODEL_NAME, messages_for_this_turn)
                end_time = time.time()

                if not completion:
                    print("API call failed, terminating file generation.")
                    break
                
                latency = end_time - start_time
                
                delta_str = ""
                if "choices" in completion and len(completion["choices"]) > 0:
                    delta_str = completion["choices"][0].get("message", {}).get("content", "")
                
                usage = completion.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                print(f"Model input: {input_str}")
                print(f"Model original output: {delta_str}")
                print(f"Time: {latency:.2f} seconds | Token consumption: input={prompt_tokens}, output={completion_tokens}, total={total_tokens}")

                cleaned_delta_str = delta_str.strip()
                if cleaned_delta_str.startswith("```json"):
                    cleaned_delta_str = cleaned_delta_str[7:]
                if cleaned_delta_str.startswith("```"):
                    cleaned_delta_str = cleaned_delta_str[3:]
                if cleaned_delta_str.endswith("```"):
                    cleaned_delta_str = cleaned_delta_str[:-3]
                
                delta_obj = json.loads(cleaned_delta_str.strip())
                
                generated_deltas.append(delta_obj)
                current_file_frame_data.append({
                    "frame": frame_num + 1, "latency_sec": latency,
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens, "is_correct": True
                })

                if is_termination_delta(delta_obj, final_line_num):
                    print("Success: model output termination signal, algorithm ended normally.")
                    break 

                current_frame = apply_delta_to_state(current_frame, delta_obj)
                last_meta.update(delta_obj.get("meta", {}))

            except Exception as e:
                print(f"Error processing frame {frame_num + 1}: {e}"); break
        
        if len(generated_deltas) == MAX_FRAMES:
            print(f"Warning: maximum frame limit ({MAX_FRAMES} frames) reached, generation forced to stop.")

        final_svl_object = {
            "svl_version": "5.0", "algorithm": algorithm_info,
            "initial_frame": initial_frame, "deltas": generated_deltas
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_svl_object, f, indent=2, ensure_ascii=False)
        print(f"Generation complete! Results saved to: {output_path}")

        if current_file_frame_data:
            successful_frames = len(current_file_frame_data)
            avg_latency = sum(d['latency_sec'] for d in current_file_frame_data) / successful_frames if successful_frames > 0 else 0
            avg_total_tokens = sum(d['total_tokens'] for d in current_file_frame_data) / successful_frames if successful_frames > 0 else 0
            total_tokens_for_file = sum(d['total_tokens'] for d in current_file_frame_data)
            
            file_stats = {
                "file_name": input_path.name, "successful_frames (FFI)": successful_frames,
                "total_tokens": total_tokens_for_file, "avg_latency_sec_per_frame": avg_latency,
                "avg_total_tokens_per_frame": avg_total_tokens
            }
            all_files_stats.append(file_stats)
            
            print("\n" + "-"*15 + " File Statistics " + "-"*15)
            print(f"File name: {file_stats['file_name']}")
            print(f"Successful frames (FFI): {file_stats['successful_frames (FFI)']}")
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

if __name__ == "__main__":
    main()