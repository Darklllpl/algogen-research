# -*- coding: utf-8 -*-
"""
Run rollout (15 files) + teacher forcing (5 manual files) under two thresholds (0.8, 0.6)
and dump EVERYTHING into one single JSON with bootstrap 95% CIs.

IMPORTANT:
1) Fill TF_FILE_LIST with EXACT 5 file names existing in INPUT_SVL_DIR.
2) (Optional) If you want to control which 15 files are used for rollout, fill ROLLOUT_FILE_LIST.
   Otherwise we'll just take the first 15 files (sorted by name).

Author: You
"""

import json
import re
from pathlib import Path
import copy
import requests
import time
import random
from statistics import mean
from typing import List, Dict, Any, Tuple
import numpy as np

# ============================
# ---------- CONFIG ----------
# ============================

# --- API ---
API_BASE_URL = 'https://az.gptplus5.com/v1'
API_TOKEN = "sk-dMdaYSQFPlEMDhGK02AeD8C2Ec0d43EdBaD8Ce0435BcC623"   
DEFAULT_MODEL = "gpt-4.1-2025-04-14"

# --- Data ---
INPUT_SVL_DIR = "svl_dataset/small_test"

TF_FILE_LIST = [
    "bubble_sort_all_same_0002.json",
    "bubble_sort_nearly_sorted_0027.json",
    "bubble_sort_random_0026.json",
    "bubble_sort_with_duplicates_0029.json",
    "insertion_sort_all_same_0017.json",
    "insertion_sort_nearly_sorted_0027.json",
    "insertion_sort_random_0028.json",
    "quicksort_all_same_0024.json",
    "quicksort_nearly_sorted_0027.json",
    "quicksort_random_0029.json"

]

ROLLOUT_FILE_LIST: List[str] = [
    # "file_a.json", ...
]

# --- Experiment ---
THRESHOLDS = [0.8, 0.6]
ROLLOUT_MAX_FILES = 15
TEACHER_FORCING_EVAL = True
MAX_FRAMES_TF = 50
MAX_FRAMES_ROLLOUT = 200
PAD_MISSING_AS_INCORRECT = True

# --- Bootstrap ---
N_BOOT = 1000
BOOT_CI = 0.95
BOOT_SEED = 42

# --- Output ---
OUTPUT_DIR = "all_results_new"
OUTPUT_JSON = "all_results_new/aaai_one_json_all_metrics.json"

# ============================
# ------ CORE LOGIC ----------
# ============================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def call_llm_api(prompt: str, model_name: str = DEFAULT_MODEL):
    """
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
        t0 = time.time()
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        t1 = time.time()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            stats = {
                "content": content,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "latency_sec": t1 - t0
            }
            return stats
        else:
            print(f"[API] Error: {result}")
            return None
    except Exception as e:
        print(f"[API] Error: {e}")
        return None

# ============================
# ---- Matching Utilities ----
# ============================

def flatten_operations(ops):
    if not isinstance(ops, list):
        return []
    result = []
    for item in ops:
        if isinstance(item, list):
            result.extend(flatten_operations(item))
        elif isinstance(item, dict):
            result.append(item)
    return result

def meta_equivalent(meta1, meta2):
    if set(meta1.keys()) != set(meta2.keys()):
        return False
    for k in meta1:
        v1, v2 = meta1[k], meta2[k]
        if v1 is None or v2 is None:
            return False
    return True

def is_delta_semantically_equal(delta1, delta2):
    if delta1.get("code_highlight") != delta2.get("code_highlight"):
        return False
    if not meta_equivalent(delta1.get("meta", {}), delta2.get("meta", {})):
        return False
    ops1 = flatten_operations(delta1.get("operations", []))
    ops2 = flatten_operations(delta2.get("operations", []))
    def op_key(op):
        return (op.get("op"), json.dumps(op.get("params", {}), sort_keys=True))
    set1 = set(map(op_key, ops1))
    set2 = set(map(op_key, ops2))
    return set1 == set2

def op_key(delta_op):
    return (
        delta_op.get("op"),
        json.dumps(delta_op.get("params", {}), sort_keys=True, ensure_ascii=False)
    )

def op_set(delta):
    ops = flatten_operations(delta.get("operations", []))
    return set(map(op_key, ops))

def op_jaccard(pred, gt):
    s1, s2 = op_set(pred), op_set(gt)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / max(1, len(s1 | s2))

def meta_hamming(meta1, meta2):
    keys = set(meta1) | set(meta2)
    return sum((meta1.get(k, "-") != meta2.get(k, "-")) for k in keys)

def normalize_delta_general(delta):
    if delta is None:
        return {}
    d = copy.deepcopy(delta)
    if "operations" in d:
        d["operations"] = flatten_operations(d.get("operations", []))
        try:
            d["operations"] = sorted(d["operations"], key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        except Exception:
            pass
    meta = d.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    d["meta"] = meta
    d["code_highlight"] = d.get("code_highlight", None)
    return d

def soft_accept(pred, gt, jacc_th=0.8, meta_hd_th=1, allow_code_off_by_one=False):
    p = normalize_delta_general(pred)
    g = normalize_delta_general(gt)

    if is_delta_semantically_equal(p, g):
        return True

    jacc = op_jaccard(p, g)
    meta_hd = meta_hamming(p.get("meta", {}), g.get("meta", {}))
    ch_p = p.get("code_highlight")
    ch_g = g.get("code_highlight")

    if allow_code_off_by_one:
        code_ok = (ch_p == ch_g) or (ch_p is not None and ch_g is not None and abs(ch_p - ch_g) == 1)
    else:
        code_ok = (ch_p == ch_g)

    return (jacc >= jacc_th) and (meta_hd <= meta_hd_th) and code_ok

def is_delta_equal(delta1, delta2):
    return json.dumps(delta1, sort_keys=True) == json.dumps(delta2, sort_keys=True)

# ============================
# --- Prompt & Extraction ----
# ============================

def build_the_prompt(user_input_str: str) -> str:
    """
    """
    prompt = '''
### ROLE & TASK
You are a 'Deterministic Algorithm Execution Machine'. Your task is to simulate the NEXT SINGLE STEP of an algorithm. To do this, you must follow a strict three-stage process: THINK, CODE, and OUTPUT.

#### Additional Rules:
- Only write a variable to meta if its value actually changes. Do not include uninitialized variables in meta.
- After a swap (CODE_LINE[6]), in the next frame, you MUST increment meta.j by 1, unless j has reached n-i-2.
- The inner loop (j) must continue until j == n-i-2, only then should you exit to the next outer loop or to return.
- Always remove temp elements (like swap_arrow) in the same frame as the swap, not in a separate frame.
- Never jump to CODE_LINE[7] (return) until all inner and outer loops are complete.
- If code_highlight would be out of range or None, fallback to the previous valid line or terminate.

#### Variable Rule Supplement:
- The “-” sentinel in meta means “inherit the value from the previous frame”, not reset or undefined.
- Only increment `j` when entering the next iteration of the inner loop (i.e., when code_highlight is the “for j ...” line). Otherwise, `j` must remain the same as in the previous frame.
- Do NOT increment the loop index (j) inside the swap step or any step except the start of the inner loop.

### STAGE 1: THINK (Chain-of-Thought)
First, analyze the provided CONTEXT. Verbalize your reasoning in a <thinking> block. You should determine what the next logical operation is based on the current state and the highlighted code line.

### STAGE 2: CODE (Python Execution)
Second, based on your reasoning, write a short, self-contained Python snippet inside a <code> block.
- This code MUST be executable.
- It will be given a dictionary named `context` as input, which contains `meta` (a dict) and `data` (a list of dicts, like `[{{"value": v, "state": s}}, ...]`).
- The code's task is to calculate the content of the *next* FrameDelta.
- It MUST create a dictionary named `result_delta` containing the calculated `meta`, `code_highlight`, and `operations`.

### STAGE 3: OUTPUT (Final JSON)
Third, copy the `result_delta` dictionary you calculated in the CODE stage into a <json_output> block. This must be a single, valid JSON object.

---
### GOLDEN EXAMPLE

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[1]: function BubbleSort(array): CONTEXT: {{"meta":{{"i":"-","j":"-"}},"data_state":{{"v":[5,3],"s":["idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
This is the initial state, not yet in the loop. The next step should jump to the for i loop.
</thinking>
<code>
meta_out = {{"i": "-", "j": "-"}}
code_highlight_out = 3
operations_out = []
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":"-","j":"-"}},"code_highlight":3,"operations":[]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[3]: for i from 0 to n-1: CONTEXT: {{"meta":{{"i":"-","j":"-"}},"data_state":{{"v":[5,3],"s":["idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
Entering the outer loop, i is initialized to 0, j is not yet assigned.
</thinking>
<code>
meta_out = {{"i": 0, "j": "-"}}
code_highlight_out = 4
operations_out = []
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":"-"}},"code_highlight":4,"operations":[]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[4]: for j from 0 to n-i-2: CONTEXT: {{"meta":{{"i":0,"j":"-"}},"data_state":{{"v":[5,3],"s":["idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
Entering the inner loop, j is initialized to 0.
</thinking>
<code>
meta_out = {{"i": 0, "j": 0}}
code_highlight_out = 5
operations_out = [
    [
        {{"op": "updateStyle", "params": {{"indices": [0], "styleKey": "comparing"}}}},
        {{"op": "updateStyle", "params": {{"indices": [1], "styleKey": "comparing"}}}}
    ]
]
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":0}},"code_highlight":5,"operations":[[{{"op":"updateStyle","params":{{"indices":[0],"styleKey":"comparing"}}}},{{"op":"updateStyle","params":{{"indices":[1],"styleKey":"comparing"}}}}]]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {{"meta":{{"i":0,"j":0}},"data_state":{{"v":[5,3],"s":["comparing","comparing"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
Comparing array[0] and array[1]. Since 5 > 3, we need to swap them. Highlight swap line and draw swap arrow.
</thinking>
<code>
meta_out = {{"i": 0, "j": 0}}
code_highlight_out = 6
operations_out = [
    [
        {{"op": "updateStyle", "params": {{"indices": [0, 1], "styleKey": "tobe_swapped"}}}},
        {{"op": "drawTemp", "params": {{"type": "swap_arrow", "indices": [0, 1], "styleKey": "swap_arrow"}}}}
    ]
]
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":0}},"code_highlight":6,"operations":[[{{"op":"updateStyle","params":{{"indices":[0,1],"styleKey":"tobe_swapped"}}}},{{"op":"drawTemp","params":{{"type":"swap_arrow","indices":[0,1],"styleKey":"swap_arrow"}}}}]]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[6]: swap(array[j], array[j+1]) CONTEXT: {{"meta":{{"i":0,"j":0}},"data_state":{{"v":[5,3],"s":["tobe_swapped","tobe_swapped"]}},"boundaries":[],"temp_elements":[{{"type":"swap_arrow","indices":[0,1],"styleKey":"swap_arrow"}}]}}

**YOUR RESPONSE:**
<thinking>
Swapping array[0] and array[1]. After swap, remove the temp arrow and set both elements to idle. j should NOT increment here, only when entering the next for j loop. Next, go back to the comparison step.
</thinking>
<code>
meta_out = {{"i": 0, "j": 0}}
code_highlight_out = 5
operations_out = [
    [
        {{"op": "removeTemp", "params": {{"type": "swap_arrow"}}}},
        {{"op": "moveElements", "params": {{"pairs": [{{"fromIndex": 0, "toIndex": 1}}, {{"fromIndex": 1, "toIndex": 0}}], "animationKey": "swap_animation"}}}},
        {{"op": "updateStyle", "params": {{"indices": [0, 1], "styleKey": "idle"}}}}
    ]
]
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":0}},"code_highlight":5,"operations":[[{{"op":"removeTemp","params":{{"type":"swap_arrow"}}}},{{"op":"moveElements","params":{{"pairs":[{{"fromIndex":0,"toIndex":1}},{{"fromIndex":1,"toIndex":0}}],"animationKey":"swap_animation"}}}},{{"op":"updateStyle","params":{{"indices":[0,1],"styleKey":"idle"}}}}]]}}
</json_output>

**USER INPUT:**
generate visual delta: ALGORITHM: Bubble Sort CODE_LINE[5]: if array[j] > array[j+1]: CONTEXT: {{"meta":{{"i":0,"j":0}},"data_state":{{"v":[3,5],"s":["idle","idle"]}},"boundaries":[],"temp_elements":[]}}

**YOUR RESPONSE:**
<thinking>
Comparing array[0] and array[1] again. Now, since 3 < 5, no swap is needed. j will increment only when entering the next for j loop.
</thinking>
<code>
meta_out = {{"i": 0, "j": 1}}
code_highlight_out = 5
operations_out = [
    [
        {{"op": "updateStyle", "params": {{"indices": [0], "styleKey": "idle"}}}},
        {{"op": "updateStyle", "params": {{"indices": [1], "styleKey": "idle"}}}}
    ]
]
result_delta = {{
    "meta": meta_out,
    "code_highlight": code_highlight_out,
    "operations": operations_out
}}
</code>
<json_output>
{{"meta":{{"i":0,"j":1}},"code_highlight":5,"operations":[[{{"op":"updateStyle","params":{{"indices":[0],"styleKey":"idle"}}}},{{"op":"updateStyle","params":{{"indices":[1],"styleKey":"idle"}}}}]]}}
</json_output>

---
### YOUR TASK

**USER INPUT:**
{user_input_str}

**YOUR RESPONSE:**
'''.format(user_input_str=user_input_str)
    return prompt

def format_input_for_cot(algorithm_info: dict, current_frame: dict, last_meta: dict, schema_var_names: list) -> str:
    algorithm_name = algorithm_info.get("name", "Unknown Algorithm")
    code_highlight_line_num = current_frame.get("code_highlight", 1)
    pseudocode_list = current_frame.get("pseudocode", [])
    if code_highlight_line_num is None or code_highlight_line_num < 1 or code_highlight_line_num > len(pseudocode_list):
        code_highlight_line_num = current_frame.get("code_highlight", 1)
    code_idx = code_highlight_line_num - 1
    code_line_text = pseudocode_list[code_idx] if 0 <= code_idx < len(pseudocode_list) else ""
    data_state_obj = current_frame.get("data_state", {})
    try:
        if isinstance(data_state_obj, dict) and "v" in data_state_obj and "s" in data_state_obj:
            compressed_data_state = data_state_obj
        elif isinstance(data_state_obj, list):
            compressed_data_state = {
                "v": [elem["value"] for elem in data_state_obj],
                "s": [elem["state"] for elem in data_state_obj]
            }
        elif isinstance(data_state_obj, dict) and "data" in data_state_obj:
            data_list = data_state_obj["data"]
            compressed_data_state = {
                "v": [elem["value"] for elem in data_list],
                "s": [elem["state"] for elem in data_list]
            }
        else:
            raise ValueError(f"Unknown data_state_obj structure: {data_state_obj}")
    except Exception as e:
        print(f"Error processing data_state_obj: {e}")
        print(f"Current data_state_obj content: {data_state_obj}")
        raise

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

def extract_delta_from_llm_output(llm_output_text: str):
    try:
        json_match = re.search(r'<json_output>(.*?)</json_output>', llm_output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            return json.loads(json_str)
        code_match = re.search(r'<code>(.*?)</code>', llm_output_text, re.DOTALL)
        if code_match:
            return None
    except Exception:
        return None
    return None

def run_single_frame_generation(user_input_str: str, model_name: str = DEFAULT_MODEL):
    prompt = build_the_prompt(user_input_str)
    api_result = call_llm_api(prompt, model_name)
    if api_result is None:
        return None, None
    llm_output_text = api_result["content"]

    delta_obj = extract_delta_from_llm_output(llm_output_text)
    if delta_obj is not None:
        stats = {
            "latency_sec": api_result["latency_sec"],
            "prompt_tokens": api_result["prompt_tokens"],
            "completion_tokens": api_result["completion_tokens"],
            "total_tokens": api_result["total_tokens"],
            "extraction_method": "json_output"
        }
        return delta_obj, stats

    try:
        code_to_exec = re.search(r'<code>(.*?)</code>', llm_output_text, re.DOTALL).group(1).strip()
    except AttributeError:
        return None, None

    # 从 user_input_str 中抽 context（我们 prompt 拼接里紧跟 CONTEXT: JSON）
    try:
        context_str = re.search(r'CONTEXT: (.*)', user_input_str).group(1)
        context_data = json.loads(context_str)
    except Exception:
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
            stats = {
                "latency_sec": api_result["latency_sec"],
                "prompt_tokens": api_result["prompt_tokens"],
                "completion_tokens": api_result["completion_tokens"],
                "total_tokens": api_result["total_tokens"],
                "extraction_method": "code_execution"
            }
            return predicted_delta, stats
        else:
            return None, None
    except Exception:
        return None, None

def apply_delta_to_state(current_frame: dict, delta: dict, last_meta: dict) -> dict:
    new_frame = copy.deepcopy(current_frame)
    data_state = new_frame["data_state"]

    if isinstance(data_state, list):
        data = data_state
    elif isinstance(data_state, dict) and "data" in data_state:
        data = data_state["data"]
    else:
        raise ValueError(f"Unknown data_state structure: {data_state}")

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

    new_frame["data_state"] = data
    return new_frame

# ============================
# --------- EVAL ------------
# ============================

def evaluate_generated_vs_gt(
    gt_deltas,
    pred_deltas,
    jacc_th=0.8,
    meta_hd_th=1,
    allow_code_off_by_one=False,
    pad_missing_as_incorrect=True
):
    n_gt = len(gt_deltas)
    n_pred = len(pred_deltas)

    frames_stats = []
    ffi_hard = None
    ffi_soft = None

    compare_len = n_gt
    for i in range(compare_len):
        if i < n_pred:
            pred = pred_deltas[i]
        else:
            if pad_missing_as_incorrect:
                pred = {}
            else:
                break

        gt = gt_deltas[i]
        hard_eq = is_delta_equal(pred, gt)
        soft_eq = soft_accept(pred, gt, jacc_th=jacc_th, meta_hd_th=meta_hd_th,
                              allow_code_off_by_one=allow_code_off_by_one)
        jacc = op_jaccard(pred, gt)

        if ffi_hard is None and not hard_eq:
            ffi_hard = i + 1
        if ffi_soft is None and not soft_eq:
            ffi_soft = i + 1

        frames_stats.append({
            "idx": i + 1,
            "hard_correct": hard_eq,
            "soft_correct": soft_eq,
            "ops_jaccard": jacc
        })

    if ffi_hard is None:
        ffi_hard = n_gt + 1
    if ffi_soft is None:
        ffi_soft = n_gt + 1

    total_frames = len(frames_stats)
    hard_correct = sum(1 for f in frames_stats if f["hard_correct"])
    soft_correct = sum(1 for f in frames_stats if f["soft_correct"])
    hard_acc = hard_correct / max(1, total_frames)
    soft_acc = soft_correct / max(1, total_frames)
    avg_jacc = sum(f["ops_jaccard"] for f in frames_stats) / max(1, total_frames)

    return {
        "ffi_hard": ffi_hard,
        "ffi_soft": ffi_soft,
        "hard_accuracy": hard_acc,
        "soft_accuracy": soft_acc,
        "avg_ops_jaccard": avg_jacc,
        "delta_ffi": ffi_soft - ffi_hard,
        "per_frame": frames_stats
    }

def teacher_forcing_eval(
    algorithm_info,
    initial_frame,
    original_deltas,
    schema_var_names,
    jacc_th,
    model_name=DEFAULT_MODEL,
    max_frames_tf=50,
    allow_code_off_by_one=False
):
    n_total = len(original_deltas)
    n = min(n_total, max_frames_tf) if max_frames_tf is not None else n_total
    truncated = (n < n_total)

    frames_stats = []
    ffi_hard_tf = None
    ffi_soft_tf = None

    total_tokens = 0
    total_latency = 0.0

    current_frame_tf = copy.deepcopy(initial_frame)
    current_frame_tf["temp_elements"] = []
    current_frame_tf["boundaries"] = []
    last_meta_tf = {}

    for i in range(n):
        user_input = format_input_for_cot(algorithm_info, current_frame_tf, last_meta_tf, schema_var_names)
        pred_delta, stats = run_single_frame_generation(user_input, model_name)
        if pred_delta is None:
            hard_eq = False
            soft_eq = False
            jacc = 0.0
            latency = 0.0
            tokens_sum = 0
        else:
            gt_delta = original_deltas[i]
            hard_eq = is_delta_equal(pred_delta, gt_delta)
            soft_eq = soft_accept(
                pred_delta, gt_delta,
                jacc_th=jacc_th,
                meta_hd_th=1,
                allow_code_off_by_one=allow_code_off_by_one
            )
            jacc = op_jaccard(pred_delta, gt_delta)
            latency = stats["latency_sec"]
            tokens_sum = stats["total_tokens"]

        if ffi_hard_tf is None and not hard_eq:
            ffi_hard_tf = i + 1
        if ffi_soft_tf is None and not soft_eq:
            ffi_soft_tf = i + 1

        frames_stats.append({
            "idx": i + 1,
            "hard_correct": hard_eq,
            "soft_correct": soft_eq,
            "ops_jaccard": jacc,
            "latency_sec": latency,
            "total_tokens": tokens_sum
        })

        total_tokens += tokens_sum
        total_latency += latency

        try:
            current_frame_tf = apply_delta_to_state(current_frame_tf, original_deltas[i], last_meta_tf)
            last_meta_tf.update(original_deltas[i].get("meta", {}))
            last_meta_tf = {k: v for k, v in last_meta_tf.items() if k in schema_var_names}
        except Exception as e:
            print(f"[TF Error] apply_delta_to_state failed: {e}")
            break

    if ffi_hard_tf is None:
        ffi_hard_tf = n + 1
    if ffi_soft_tf is None:
        ffi_soft_tf = n + 1

    total_frames = len(frames_stats)
    hard_correct = sum(1 for f in frames_stats if f["hard_correct"])
    soft_correct = sum(1 for f in frames_stats if f["soft_correct"])
    step_hard_acc = hard_correct / max(1, total_frames)
    step_soft_acc = soft_correct / max(1, total_frames)
    avg_jacc = mean([f["ops_jaccard"] for f in frames_stats]) if frames_stats else 0.0
    avg_latency = total_latency / max(1, total_frames)

    return {
        "step_hard_acc": step_hard_acc,
        "step_soft_acc": step_soft_acc,
        "ffi_hard_tf": ffi_hard_tf,
        "ffi_soft_tf": ffi_soft_tf,
        "delta_ffi_tf": ffi_soft_tf - ffi_hard_tf,
        "avg_ops_jaccard": avg_jacc,
        "frames": frames_stats,
        "total_tokens": total_tokens,
        "avg_latency_sec": avg_latency,
        "tf_frames_evaluated": total_frames,
        "tf_truncated": truncated,
        "tf_coverage": total_frames / max(1, n_total)
    }

def ci_low_high(samples: List[float], ci: float = 0.95) -> Tuple[float, float]:
    lower = (1 - ci) / 2 * 100
    upper = (1 + ci) / 2 * 100
    return (float(np.percentile(samples, lower)), float(np.percentile(samples, upper)))

def bootstrap_macro_ci(values: List[float], n_boot=N_BOOT, ci=BOOT_CI, seed=BOOT_SEED) -> Dict[str, Any]:
    if not values:
        return {"mean": 0.0, "ci": [0.0, 0.0]}
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means.append(values[idx].mean())
    low, high = ci_low_high(means, ci)
    return {"mean": float(values.mean()), "ci": [low, high]}

def compute_rollout_micro_macro_and_ci(per_file: List[Dict[str, Any]], n_boot=N_BOOT, ci=BOOT_CI, seed=BOOT_SEED):
    if not per_file:
        return {"micro": {}, "macro": {}, "micro_ci": {}, "macro_ci": {}}

    # macro
    macro_fields = [
        "hard_accuracy", "soft_accuracy",
        "ffi_hard", "ffi_soft", "delta_ffi",
        "avg_ops_jaccard"
    ]
    macro_collect = {k: [] for k in macro_fields}
    for r in per_file:
        macro_collect["hard_accuracy"].append(r["hard_accuracy"])
        macro_collect["soft_accuracy"].append(r["soft_accuracy"])
        macro_collect["ffi_hard"].append(r["ffi_hard"])
        macro_collect["ffi_soft"].append(r["ffi_soft"])
        macro_collect["delta_ffi"].append(r["delta_ffi"])
        macro_collect["avg_ops_jaccard"].append(r["avg_ops_jaccard"])

    macro = {k: float(mean(v)) if v else 0.0 for k, v in macro_collect.items()}
    macro_ci = {k: bootstrap_macro_ci(macro_collect[k], n_boot=n_boot, ci=ci, seed=seed) for k in macro_fields}

    # micro
    total_frames = sum(len(r["per_frame"]) for r in per_file)
    hard_correct = sum(sum(1 for f in r["per_frame"] if f["hard_correct"]) for r in per_file)
    soft_correct = sum(sum(1 for f in r["per_frame"] if f["soft_correct"]) for r in per_file)
    avg_ffi_hard = mean([r["ffi_hard"] for r in per_file])
    avg_ffi_soft = mean([r["ffi_soft"] for r in per_file])
    avg_delta_ffi = mean([r["delta_ffi"] for r in per_file])
    avg_jacc = mean([r["avg_ops_jaccard"] for r in per_file])

    micro = {
        "hard_accuracy": hard_correct / max(1, total_frames),
        "soft_accuracy": soft_correct / max(1, total_frames),
        "avg_ffi_hard": avg_ffi_hard,
        "avg_ffi_soft": avg_ffi_soft,
        "avg_delta_ffi": avg_delta_ffi,
        "avg_ops_jaccard": avg_jacc
    }

    # bootstrap micro (approx by resampling files)
    rng = np.random.default_rng(seed)
    micro_boot = {k: [] for k in micro.keys()}
    n_files = len(per_file)

    for _ in range(n_boot):
        idx = rng.integers(0, n_files, size=n_files)
        sel = [per_file[i] for i in idx]
        total_frames_bs = sum(len(r["per_frame"]) for r in sel)
        hard_correct_bs = sum(sum(1 for f in r["per_frame"] if f["hard_correct"]) for r in sel)
        soft_correct_bs = sum(sum(1 for f in r["per_frame"] if f["soft_correct"]) for r in sel)
        avg_ffi_hard_bs = mean([r["ffi_hard"] for r in sel])
        avg_ffi_soft_bs = mean([r["ffi_soft"] for r in sel])
        avg_delta_ffi_bs = mean([r["delta_ffi"] for r in sel])
        avg_jacc_bs = mean([r["avg_ops_jaccard"] for r in sel])

        micro_boot["hard_accuracy"].append(hard_correct_bs / max(1, total_frames_bs))
        micro_boot["soft_accuracy"].append(soft_correct_bs / max(1, total_frames_bs))
        micro_boot["avg_ffi_hard"].append(avg_ffi_hard_bs)
        micro_boot["avg_ffi_soft"].append(avg_ffi_soft_bs)
        micro_boot["avg_delta_ffi"].append(avg_delta_ffi_bs)
        micro_boot["avg_ops_jaccard"].append(avg_jacc_bs)

    micro_ci = {}
    for k, arr in micro_boot.items():
        low, high = ci_low_high(arr, ci)
        micro_ci[k] = {"mean": micro[k], "ci": [low, high]}

    return {"micro": micro, "macro": macro, "micro_ci": micro_ci, "macro_ci": macro_ci}

def compute_tf_micro_macro_and_ci(per_file: List[Dict[str, Any]], n_boot=N_BOOT, ci=BOOT_CI, seed=BOOT_SEED):
    if not per_file:
        return {"micro": {}, "macro": {}, "micro_ci": {}, "macro_ci": {}}

    macro_fields = [
        "step_hard_acc", "step_soft_acc",
        "ffi_hard_tf", "ffi_soft_tf", "delta_ffi_tf",
        "avg_ops_jaccard"
    ]
    macro_collect = {k: [] for k in macro_fields}
    for r in per_file:
        macro_collect["step_hard_acc"].append(r["step_hard_acc"])
        macro_collect["step_soft_acc"].append(r["step_soft_acc"])
        macro_collect["ffi_hard_tf"].append(r["ffi_hard_tf"])
        macro_collect["ffi_soft_tf"].append(r["ffi_soft_tf"])
        macro_collect["delta_ffi_tf"].append(r["delta_ffi_tf"])
        macro_collect["avg_ops_jaccard"].append(r["avg_ops_jaccard"])

    macro = {k: float(mean(v)) if v else 0.0 for k, v in macro_collect.items()}
    macro_ci = {k: bootstrap_macro_ci(macro_collect[k], n_boot=n_boot, ci=ci, seed=seed) for k in macro_fields}

    total_frames = sum(len(r["frames"]) for r in per_file)
    hard_correct = sum(sum(1 for f in r["frames"] if f["hard_correct"]) for r in per_file)
    soft_correct = sum(sum(1 for f in r["frames"] if f["soft_correct"]) for r in per_file)
    avg_ffi_hard_tf = mean([r["ffi_hard_tf"] for r in per_file])
    avg_ffi_soft_tf = mean([r["ffi_soft_tf"] for r in per_file])
    avg_delta_ffi_tf = mean([r["delta_ffi_tf"] for r in per_file])
    avg_jacc_tf = mean([r["avg_ops_jaccard"] for r in per_file])

    micro = {
        "step_hard_accuracy": hard_correct / max(1, total_frames),
        "step_soft_accuracy": soft_correct / max(1, total_frames),
        "avg_ffi_hard_tf": avg_ffi_hard_tf,
        "avg_ffi_soft_tf": avg_ffi_soft_tf,
        "avg_delta_ffi_tf": avg_delta_ffi_tf,
        "avg_ops_jaccard_tf": avg_jacc_tf
    }

    rng = np.random.default_rng(seed)
    micro_boot = {k: [] for k in micro.keys()}
    n_files = len(per_file)

    for _ in range(n_boot):
        idx = rng.integers(0, n_files, size=n_files)
        sel = [per_file[i] for i in idx]
        total_frames_bs = sum(len(r["frames"]) for r in sel)
        hard_correct_bs = sum(sum(1 for f in r["frames"] if f["hard_correct"]) for r in sel)
        soft_correct_bs = sum(sum(1 for f in r["frames"] if f["soft_correct"]) for r in sel)
        avg_ffi_hard_tf_bs = mean([r["ffi_hard_tf"] for r in sel])
        avg_ffi_soft_tf_bs = mean([r["ffi_soft_tf"] for r in sel])
        avg_delta_ffi_tf_bs = mean([r["delta_ffi_tf"] for r in sel])
        avg_jacc_tf_bs = mean([r["avg_ops_jaccard"] for r in sel])

        micro_boot["step_hard_accuracy"].append(hard_correct_bs / max(1, total_frames_bs))
        micro_boot["step_soft_accuracy"].append(soft_correct_bs / max(1, total_frames_bs))
        micro_boot["avg_ffi_hard_tf"].append(avg_ffi_hard_tf_bs)
        micro_boot["avg_ffi_soft_tf"].append(avg_ffi_soft_tf_bs)
        micro_boot["avg_delta_ffi_tf"].append(avg_delta_ffi_tf_bs)
        micro_boot["avg_ops_jaccard_tf"].append(avg_jacc_tf_bs)

    micro_ci = {}
    for k, arr in micro_boot.items():
        low, high = ci_low_high(arr, ci)
        micro_ci[k] = {"mean": micro[k], "ci": [low, high]}

    return {"micro": micro, "macro": macro, "micro_ci": micro_ci, "macro_ci": macro_ci}

# ============================
# ---- Run per threshold -----
# ============================

def rollout_one_file(
    input_path: Path,
    output_dir: Path,
    jacc_th: float
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate (rollout) and offline evaluate (hard/soft), return:
    - rollout_per_file_eval structure (evaluate_generated_vs_gt return + pred_only/coverage etc.)
    - low-level statistics (tokens/latency etc.) file_stats_raw
    """
    svl_data = json.loads(input_path.read_text(encoding='utf-8'))
    algorithm_info = svl_data["algorithm"]
    initial_frame = svl_data["initial_frame"]
    original_deltas = svl_data.get("deltas", [])

    schema_var_names = [var['name'] for var in initial_frame.get("variables_schema", [])]

    current_frame = copy.deepcopy(initial_frame)
    current_frame["temp_elements"] = []
    current_frame["boundaries"] = []
    last_meta = {}

    generated_deltas = []
    frame_stats_list = []

    ffi_hard_local = None
    ffi_soft_local = None

    for frame_num in range(MAX_FRAMES_ROLLOUT):
        user_input = format_input_for_cot(algorithm_info, current_frame, last_meta, schema_var_names)
        result = run_single_frame_generation(user_input, DEFAULT_MODEL)
        if result is None or result[0] is None:
            if result and result[0] is not None:
                delta_obj, stats = result
                generated_deltas.append(delta_obj)
            break

        delta_obj, stats = result
        frame_stats_list.append(stats)

        if frame_num < len(original_deltas):
            gt_delta = original_deltas[frame_num]
            hard_eq = is_delta_equal(delta_obj, gt_delta)
            soft_eq = soft_accept(
                delta_obj, gt_delta,    
                jacc_th=jacc_th,
                meta_hd_th=1,
                allow_code_off_by_one=False
            )
            if ffi_hard_local is None and not hard_eq:
                ffi_hard_local = frame_num + 1
            if ffi_soft_local is None and not soft_eq:
                ffi_soft_local = frame_num + 1
            if not soft_eq:
                generated_deltas.append(delta_obj)
                break

        generated_deltas.append(delta_obj)

        try:
            current_frame = apply_delta_to_state(current_frame, delta_obj, last_meta)
            last_meta.update(delta_obj.get("meta", {}))
            last_meta = {k: v for k, v in last_meta.items() if k in schema_var_names}
        except Exception as e:
            print(f"[Rollout Error] apply_delta_to_state failed: {e}")
            break

    if ffi_hard_local is None:
        ffi_hard_local = len(original_deltas) + 1
    if ffi_soft_local is None:
        ffi_soft_local = len(original_deltas) + 1

    out_svl = {
        "svl_version": "5.0",
        "algorithm": algorithm_info,
        "initial_frame": initial_frame,
        "deltas": generated_deltas
    }
    output_path = output_dir / input_path.name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out_svl, f, indent=2, ensure_ascii=False)

    rollout_eval = evaluate_generated_vs_gt(
        gt_deltas=original_deltas,
        pred_deltas=generated_deltas,
        jacc_th=jacc_th,
        meta_hd_th=1,
        allow_code_off_by_one=False,
        pad_missing_as_incorrect=PAD_MISSING_AS_INCORRECT
    )
    rollout_eval["file_name"] = input_path.name
    rollout_eval["ffi_hard_rollout_online"] = ffi_hard_local
    rollout_eval["ffi_soft_rollout_online"] = ffi_soft_local

    n_pred = len(generated_deltas)
    hard_correct_in_pred = 0
    soft_correct_in_pred = 0
    limit = min(n_pred, len(original_deltas))
    for i in range(limit):
        if is_delta_equal(generated_deltas[i], original_deltas[i]):
            hard_correct_in_pred += 1
        if soft_accept(
            generated_deltas[i], original_deltas[i],
            jacc_th=jacc_th, meta_hd_th=1, allow_code_off_by_one=False
        ):
            soft_correct_in_pred += 1
    hard_acc_pred_only = hard_correct_in_pred / max(1, n_pred)
    soft_acc_pred_only = soft_correct_in_pred / max(1, n_pred)
    coverage = n_pred / max(1, len(original_deltas))

    rollout_eval["hard_acc_pred_only"] = hard_acc_pred_only
    rollout_eval["soft_acc_pred_only"] = soft_acc_pred_only
    rollout_eval["coverage"] = coverage

    successful_frames = len(frame_stats_list)
    avg_latency = (sum(d['latency_sec'] for d in frame_stats_list) / successful_frames) if successful_frames > 0 else 0.0
    avg_total_tokens = (sum(d['total_tokens'] for d in frame_stats_list) / successful_frames) if successful_frames > 0 else 0.0
    total_tokens_for_file = sum(d['total_tokens'] for d in frame_stats_list)
    raw_stats = {
        "file_name": input_path.name,
        "successful_frames (FFI_rollout_len)": successful_frames,
        "total_tokens_rollout": total_tokens_for_file,
        "avg_latency_sec_per_frame_rollout": avg_latency,
        "avg_total_tokens_per_frame_rollout": avg_total_tokens
    }

    return rollout_eval, raw_stats

def tf_one_file(
    input_path: Path,
    jacc_th: float
) -> Dict[str, Any]:
    svl_data = json.loads(input_path.read_text(encoding='utf-8'))
    algorithm_info = svl_data["algorithm"]
    initial_frame = svl_data["initial_frame"]
    original_deltas = svl_data.get("deltas", [])
    schema_var_names = [var['name'] for var in initial_frame.get("variables_schema", [])]

    tf_result = teacher_forcing_eval(
        algorithm_info=algorithm_info,
        initial_frame=initial_frame,
        original_deltas=original_deltas,
        schema_var_names=schema_var_names,
        jacc_th=jacc_th,
        model_name=DEFAULT_MODEL,
        max_frames_tf=MAX_FRAMES_TF,
        allow_code_off_by_one=False
    )
    tf_result["file_name"] = input_path.name
    return tf_result

def run_for_threshold(
    threshold: float,
    rollout_files: List[Path],
    tf_files: List[Path],
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run rollout + TF for a threshold, return structure that can be put into results[th_str]
    """
    th_dir = output_dir / f"th_{threshold}"
    ensure_dir(th_dir)

    rollout_per_file = []
    rollout_raw_stats = []
    for p in rollout_files:
        print(f"\n[TH={threshold}] Rollout on: {p.name}")
        r_eval, r_raw = rollout_one_file(p, th_dir, jacc_th=threshold)
        rollout_per_file.append(r_eval)
        rollout_raw_stats.append(r_raw)

    rollout_summary = compute_rollout_micro_macro_and_ci(rollout_per_file, n_boot=N_BOOT, ci=BOOT_CI, seed=BOOT_SEED)

    tf_per_file = []
    tf_summary = {}
    if TEACHER_FORCING_EVAL:
        for p in tf_files:
            print(f"\n[TH={threshold}] TF on: {p.name}")
            tf_res = tf_one_file(p, threshold)
            tf_per_file.append(tf_res)
        tf_summary = compute_tf_micro_macro_and_ci(tf_per_file, n_boot=N_BOOT, ci=BOOT_CI, seed=BOOT_SEED)

    return {
        "rollout": {
            "per_file": rollout_per_file,
            "summary": rollout_summary,
            "raw_stats": rollout_raw_stats
        },
        "teacher_forcing": {
            "per_file": tf_per_file,
            "summary": tf_summary
        }
    }

def diff_macro_summary(res_a: Dict[str, Any], res_b: Dict[str, Any], path: List[str], metric_keys: List[str]) -> Dict[str, float]:
    """
    Calculate macro mean diff of res_b - res_a under given path
    path example: ["rollout", "summary", "macro"]
    """
    d = {}
    # navigate
    def get_by_path(dct, path):
        x = dct
        for k in path:
            x = x[k]
        return x
    a = get_by_path(res_a, path)
    b = get_by_path(res_b, path)
    for k in metric_keys:
        d[k] = float(b.get(k, 0.0) - a.get(k, 0.0))
    return d

# ============================
# ----------- MAIN -----------
# ============================

def main():
    input_dir = Path(INPUT_SVL_DIR)
    all_jsons = sorted([p for p in input_dir.glob("*.json")])
    all_names = [p.name for p in all_jsons]

    if not TF_FILE_LIST:
        raise ValueError("Please manually list 5 sample file names in TF_FILE_LIST (must exist in INPUT_SVL_DIR)!")
    if any(name not in all_names for name in TF_FILE_LIST):
        missing = [name for name in TF_FILE_LIST if name not in all_names]
        raise ValueError(f"The following files in TF_FILE_LIST do not exist in {INPUT_SVL_DIR}: {missing}")
    tf_files = [input_dir / name for name in TF_FILE_LIST]

    # rollout files
    if ROLLOUT_FILE_LIST:
        if any(name not in all_names for name in ROLLOUT_FILE_LIST):
            missing = [name for name in ROLLOUT_FILE_LIST if name not in all_names]
            raise ValueError(f"The following files in ROLLOUT_FILE_LIST do not exist in {INPUT_SVL_DIR}: {missing}")
        rollout_files = [input_dir / name for name in ROLLOUT_FILE_LIST]
    else:
        if len(all_jsons) < ROLLOUT_MAX_FILES:
            print(f"[WARN] There are only {len(all_jsons)} json files, less than ROLLOUT_MAX_FILES={ROLLOUT_MAX_FILES}, all will be used.")
            rollout_files = all_jsons
        else:
            rollout_files = all_jsons[:ROLLOUT_MAX_FILES]

    ensure_dir(Path(OUTPUT_DIR))

    out = {
        "meta": {
            "created_at": now_utc_iso(),
            "model": DEFAULT_MODEL,
            "rollout_sample_n": len(rollout_files),
            "tf_sample_n": len(tf_files),
            "tf_max_frames": MAX_FRAMES_TF,
            "pad_missing_as_incorrect": PAD_MISSING_AS_INCORRECT,
            "n_bootstrap": N_BOOT,
            "bootstrap_ci": BOOT_CI
        },
        "thresholds": THRESHOLDS,
        "rollout_files": [p.name for p in rollout_files],
        "tf_files": [p.name for p in tf_files],
        "results": {}
    }

    results_by_th = {}
    for th in THRESHOLDS:
        print(f"\n========== Running threshold={th} ==========")
        res = run_for_threshold(
            threshold=th,
            rollout_files=rollout_files,
            tf_files=tf_files,
            output_dir=Path(OUTPUT_DIR)
        )
        results_by_th[str(th)] = res

    out["results"] = results_by_th

    if len(THRESHOLDS) >= 2:
        th_a = str(THRESHOLDS[0])
        th_b = str(THRESHOLDS[1])
        abl = {
            "rollout_macro_diff": diff_macro_summary(
                results_by_th[th_a], results_by_th[th_b],
                ["rollout", "summary", "macro"],
                metric_keys=["hard_accuracy", "soft_accuracy", "ffi_hard", "ffi_soft", "delta_ffi", "avg_ops_jaccard"]
            ),
            "tf_macro_diff": diff_macro_summary(
                results_by_th[th_a], results_by_th[th_b],
                ["teacher_forcing", "summary", "macro"],
                metric_keys=["step_hard_acc", "step_soft_acc", "ffi_hard_tf", "ffi_soft_tf", "delta_ffi_tf", "avg_ops_jaccard"]
            )
        }
        out["ablation"] = abl

    ensure_dir(Path(OUTPUT_JSON).parent)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Single JSON written to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
