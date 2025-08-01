import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

def apply_delta_to_visual_state(visual_state, delta):
    """
    Apply one delta to the current visual state and return the new state.
    Adjusted for SVL 4.0 insertion sort deltas.
    """
    state = copy.deepcopy(visual_state)
    meta = delta.get('meta', {})
    state['meta'].update(meta)
    state['code_line_num'] = delta.get('code_highlight', state.get('code_line_num'))
    state['temp_elements'].clear()

    for op_group in delta.get('operations', []):
        ops = op_group if isinstance(op_group, list) else [op_group]

        for op in ops:
            name = op.get('op')
            params = op.get('params', {})

            if name == 'updateStyle':
                idxs, style = params.get('indices', []), params.get('styleKey')
                if style:
                    for i in idxs:
                        if 0 <= i < len(state['data_state']):
                            state['data_state'][i]['state'] = style

            elif name == 'moveElements':
                pairs = params.get('pairs', [])
                snap = [e.copy() for e in state['data_state']]
                for p in pairs:
                    fr, to = p['fromIndex'], p['toIndex']
                    if 0 <= fr < len(snap) and 0 <= to < len(state['data_state']):
                        moved_element = snap[fr]
                        state['data_state'][to] = {
                            'index': to,
                            'value': moved_element['value'],
                            'state': moved_element['state']
                        }
                    else:
                        print(f"WARN: moveElements - Index out of bounds: from {fr}, to {to} (len: {len(state['data_state'])})")

            elif name == 'shiftElements':
                shifts = params.get('shifts', [])
                snap = [e.copy() for e in state['data_state']]
                for s in shifts:
                    fr, to, val = s.get('fromIndex'), s.get('toIndex'), s.get('value')
                    if fr is not None and to is not None and 0 <= fr < len(snap) and 0 <= to < len(state['data_state']):
                        state['data_state'][to] = {
                            'index': to,
                            'value': snap[fr]['value'],
                            'state': snap[fr]['state']
                        }
                    else:
                        print(f"WARN: shiftElements - Index or value invalid: from {fr}, to {to}, value {val} (len: {len(state['data_state'])})")

            elif name == 'updateValues':
                updates = params.get('updates', [])
                snap = [e.copy() for e in state['data_state']]
                for u in updates:
                    idx, val = u.get('index'), u.get('value')
                    if idx is None:
                        continue
                    if 0 <= idx < len(state['data_state']):
                        state['data_state'][idx] = {
                            'index': idx,
                            'value': val,
                            'state': snap[idx]['state']
                        }
                    else:
                        print(f"WARN: updateValues - Index out of bounds: {idx} (len: {len(state['data_state'])})")

            elif name == 'updateBoundary':
                t = params.get('type')
                state['boundaries'] = [b for b in state['boundaries'] if b.get('type') != t]
                state['boundaries'].append(params)

            elif name == 'removeBoundary':
                t = params.get('type')
                state['boundaries'] = [b for b in state['boundaries'] if b.get('type') != t]

            elif name == 'drawTemp':
                state['temp_elements'].append(params)

            elif name == 'removeTemp':
                t = params.get('type')
                state['temp_elements'] = [e for e in state['temp_elements'] if e.get('type') != t]

    for i, el in enumerate(state['data_state']):
        el['index'] = i

    return state

def _get_compact_data_state(data_state_list):
    """Convert data_state to compact format: {'v':[...], 's':[...]}."""
    return {
        "v": [e['value'] for e in data_state_list],
        "s": [e['state'] for e in data_state_list]
    }

def parse_svl4_file_to_t5_pairs(svl_path):
    """
    Parse SVL file and return (input, target) pairs.
    """
    svl = json.load(open(svl_path, encoding='utf-8'))
    algo = svl.get('algorithm', {}).get('name', 'Unknown')
    init = svl['initial_frame']
    deltas = svl['deltas']
    schema = init.get('variables_schema', [])
    pseudo = init.get('pseudocode', [])

    state = {
        "meta": {v['name']:'-' for v in schema},
        "code_line_num": init.get('code_highlight', 1),
        "data_state": copy.deepcopy(init['data_state']),
        "boundaries": [],
        "temp_elements": []
    }

    pairs = []
    for d in deltas:
        compact = _get_compact_data_state(state['data_state'])
        ctx = json.dumps({
            "meta": state['meta'],
            "data_state": compact,
            "boundaries": state['boundaries'],
            "temp_elements": state['temp_elements']
        }, separators=(',', ':'))
        ln = state['code_line_num']
        code = pseudo[ln-1].strip() if 1 <= ln <= len(pseudo) else ""
        inp = f"generate visual delta: ALGORITHM: {algo} CODE_LINE[{ln}]: {code} CONTEXT: {ctx}"
        tgt = json.dumps(d, separators=(',', ':'))
        pairs.append({"input": inp, "target": tgt})
        state = apply_delta_to_visual_state(state, d)

    return pairs

def verify_dataset_file(fp):
    """Verify output JSONL format and content."""
    print(f"\nVerifying file: {fp}")
    ok, cnt = True, 0
    for i, line in enumerate(open(fp, encoding='utf-8')):
        cnt += 1
        try:
            jd = json.loads(line)
            assert 'input' in jd and 'target' in jd
            json.loads(jd['target'])
        except Exception as e:
            print(f"Line {i+1} error: {e}")
            ok = False
            break
    if ok:
        print(f"✅ Verification passed, {cnt} lines.")
    return ok

def main(args):
    inp = Path(args.input_dir)
    out = Path(args.output_file)
    files = sorted(inp.rglob("*.json"))
    all_pairs = []
    for f in tqdm(files, desc="Processing SVL files"):
        all_pairs.extend(parse_svl4_file_to_t5_pairs(f))

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as fw:
        for p in all_pairs:
            fw.write(json.dumps(p, ensure_ascii=False) + "\n")

    verify_dataset_file(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with SVL 4.0 JSON files")
    parser.add_argument("--output_file", default="train_dataset.jsonl", help="Output JSONL file path")
    args = parser.parse_args()
    main(args)
