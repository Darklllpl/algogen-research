import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

def apply_delta_to_visual_state(visual_state, delta):
    """Apply delta to visual state and return the result."""
    state = copy.deepcopy(visual_state)

    state["meta"].update(delta.get("meta", {}))
    state["code_line_num"] = delta.get("code_highlight", state.get("code_line_num"))

    state["temp_elements"].clear()

    for op_group in delta.get("operations", []):
        ops = op_group if isinstance(op_group, list) else [op_group]
        for op in ops:
            name = op.get("op")
            params = op.get("params", {})

            if name == "updateStyle":
                idxs, style = params.get("indices", []), params.get("styleKey")
                if style:
                    for i in idxs:
                        if 0 <= i < len(state["data_state"]):
                            state["data_state"][i]["state"] = style

            elif name == "moveElements":
                pairs = params.get("pairs", [])
                snap = [e.copy() for e in state["data_state"]]
                for p in pairs:
                    fr, to = p.get("fromIndex"), p.get("toIndex")
                    if (
                        fr is not None
                        and to is not None
                        and 0 <= fr < len(snap)
                        and 0 <= to < len(state["data_state"])
                    ):
                        moved = snap[fr]
                        state["data_state"][to] = {
                            "index": to,
                            "value": moved["value"],
                            "state": moved["state"],
                        }

            elif name == "shiftElements":
                shifts = params.get("shifts", [])
                snap = [e.copy() for e in state["data_state"]]
                for s in shifts:
                    fr, to = s.get("fromIndex"), s.get("toIndex")
                    if (
                        fr is not None
                        and to is not None
                        and 0 <= fr < len(snap)
                        and 0 <= to < len(state["data_state"])
                    ):
                        state["data_state"][to] = {
                            "index": to,
                            "value": snap[fr]["value"],
                            "state": snap[fr]["state"],
                        }

            elif name == "updateValues":
                updates = params.get("updates", [])
                snap = [e.copy() for e in state["data_state"]]
                for u in updates:
                    idx, val = u.get("index"), u.get("value")
                    if idx is not None and 0 <= idx < len(state["data_state"]):
                        state["data_state"][idx] = {
                            "index": idx,
                            "value": val,
                            "state": snap[idx]["state"],
                        }

            elif name == "updateBoundary":
                t = params.get("type")
                state["boundaries"] = [b for b in state["boundaries"] if b.get("type") != t]
                state["boundaries"].append(params)

            elif name == "removeBoundary":
                t = params.get("type")
                state["boundaries"] = [b for b in state["boundaries"] if b.get("type") != t]

            elif name == "drawTemp":
                state["temp_elements"].append(params)

            elif name == "removeTemp":
                t = params.get("type")
                state["temp_elements"] = [e for e in state["temp_elements"] if e.get("type") != t]

def _get_compact_data_state(data_state_list):
    """Return compressed data_state."""
    return {
        "v": [e["value"] for e in data_state_list],
        "s": [e["state"] for e in data_state_list],
    }

def create_gemma_training_example(input_str, target_str):
    """Format one Gemma training example."""
    return {
        "text": (
            "<start_of_turn>user\n"
            f"{input_str}<end_of_turn>\n"
            "<start_of_turn>model\n"
            f"{target_str}<eos>"
        )
    }

def parse_svl4_file_to_gemma_examples(svl_filepath):
    """Convert SVL file to Gemma training examples."""
    with open(svl_filepath, "r", encoding="utf-8") as f:
        svl = json.load(f)

    algo_name = svl.get("algorithm", {}).get("name", "Unknown Algorithm")
    init, deltas = svl["initial_frame"], svl["deltas"]
    schema, pseudo = init.get("variables_schema", []), init.get("pseudocode", [])

    state = {
        "meta": {v["name"]: "-" for v in schema},
        "code_line_num": init.get("code_highlight", 1),
        "data_state": copy.deepcopy(init["data_state"]),
        "boundaries": [],
        "temp_elements": [],
    }

    examples = []
    for d in deltas:
        ctx = json.dumps(
            {
                "meta": state["meta"],
                "data_state": _get_compact_data_state(state["data_state"]),
                "boundaries": state["boundaries"],
                "temp_elements": state["temp_elements"],
            },
            separators=(",", ":"),
        )
        ln = state["code_line_num"]
        code = pseudo[ln - 1].strip() if 1 <= ln <= len(pseudo) else ""
        inp = f"generate visual delta: ALGORITHM: {algo_name} CODE_LINE[{ln}]: {code} CONTEXT: {ctx}"
        tgt = json.dumps(d, separators=(",", ":"))
        examples.append(create_gemma_training_example(inp, tgt))

        state = apply_delta_to_visual_state(state, d)

    return examples

def verify_dataset_file(filepath):
    """Check dataset file format."""
    print(f"\nVerifying generated file: {filepath}")
    ok, cnt = True, 0
    for i, line in enumerate(open(filepath, encoding="utf-8")):
        cnt += 1
        try:
            jd = json.loads(line)
            assert "text" in jd
            txt = jd["text"]
            assert "<start_of_turn>user" in txt and "<start_of_turn>model" in txt and "<eos>" in txt
        except Exception as e:
            print(f"❌ Line {i+1} error: {e}")
            ok = False
            break
    if ok:
        print(f"✅ File verification passed, {cnt} lines.")
    return ok

def main(args):
    """Main process for SVL to Gemma dataset conversion."""
    inp_dir, out_file = Path(args.input_dir), Path(args.output_file)
    if not inp_dir.is_dir():
        print(f"Error: {inp_dir} is not a valid directory"); return
    svl_files = sorted(inp_dir.rglob("*.json"))
    if not svl_files:
        print("Error: No .json files found"); return

    all_examples = []
    for f in tqdm(svl_files, desc="Processing SVL files"):
        try:
            all_examples.extend(parse_svl4_file_to_gemma_examples(f))
        except Exception as e:
            print(f"\nException when processing {f.name}: {e}")
            import traceback; traceback.print_exc()

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fw:
        for ex in all_examples:
            fw.write(json.dumps(ex, ensure_ascii=False) + "\n")

    verify_dataset_file(out_file)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert SVL 4.0 to Gemma conversation format dataset")
    p.add_argument("--input_dir", required=True, help="SVL 4.0 JSON files directory")
    p.add_argument("--output_file", default="s2s_data/gemma_train_dataset.jsonl")
    main(p.parse_args())
