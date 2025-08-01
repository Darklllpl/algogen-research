import json
import copy
import argparse
from pathlib import Path
from tqdm import tqdm
import sys

# --------- Improved: Robust apply_delta_to_visual_state ----------
def apply_delta_to_visual_state(visual_state, delta):
    """
    Consistent with create_t5_native_dataset.py.
    Handles shiftElements, updateValues, and other array changes.
    """
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

    for i, e in enumerate(state["data_state"]):
        e["index"] = i

    return state
# ---------------------------------------------------------------

def _get_compact_data_state(ds):
    """Return compact data_state dict."""
    return {"v": [e["value"] for e in ds], "s": [e["state"] for e in ds]}

# ------------- Parse SVL to (X, Y_snapshot) pairs ---------------
def parse_svl4_file_to_t5_pairs_snapshot(svl_path):
    """Parse SVL file and return (input, target) snapshot pairs."""
    svl = json.load(open(svl_path, encoding="utf-8"))
    algo = svl.get("algorithm", {}).get("name", "Unknown")
    init, deltas = svl["initial_frame"], svl["deltas"]
    schema, pseudo = init.get("variables_schema", []), init.get("pseudocode", [])

    state = {
        "meta": {v["name"]: "-" for v in schema},
        "code_line_num": init.get("code_highlight", 1),
        "data_state": copy.deepcopy(init["data_state"]),
        "boundaries": [],
        "temp_elements": [],
    }

    pairs = []
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
        X = f"generate visual delta: ALGORITHM: {algo} CODE_LINE[{ln}]: {code} CONTEXT: {ctx}"

        # Target Y: next frame's compact array snapshot
        next_state = apply_delta_to_visual_state(state, d)
        Y = json.dumps(_get_compact_data_state(next_state["data_state"]), separators=(",", ":"))

        pairs.append({"input": X, "target": Y})
        state = next_state

    return pairs
# ---------------------------------------------------------------

def verify_dataset_file(fp):
    """Verify dataset file format."""
    print(f"\nVerifying file: {fp}")
    ok, cnt = True, 0
    for i, line in enumerate(open(fp, encoding="utf-8")):
        cnt += 1
        try:
            jd = json.loads(line)
            assert "input" in jd and "target" in jd
            json.loads(jd["target"])
        except Exception as e:
            print(f"❌ Line {i+1} error: {e}")
            ok = False
            break
    if ok:
        print(f"✅ Verification passed, {cnt} lines.")
    return ok

def main(args):
    """Main function for SVL to T5 snapshot dataset."""
    inp, out = Path(args.input_dir), Path(args.output_file)
    if not inp.is_dir():
        print(f"Error: {inp} is not a valid directory"); return
    files = sorted(inp.rglob("*.json"))
    if not files:
        print("Error: No .json files found"); return

    all_pairs = []
    for f in tqdm(files, desc="Processing SVL files (snapshot)"):
        try:
            all_pairs.extend(parse_svl4_file_to_t5_pairs_snapshot(f))
        except Exception as e:
            print(f"\nException in {f.name}: {e}")
            import traceback; traceback.print_exc()

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fw:
        for p in all_pairs:
            fw.write(json.dumps(p, ensure_ascii=False) + "\n")

    verify_dataset_file(out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SVL to T5 snapshot dataset generator")
    p.add_argument("--input_dir", required=True, help="SVL 4.0 JSON directory")
    p.add_argument("--output_file", default="s2s_data_snapshot/train_dataset_snapshot.jsonl")
    main(p.parse_args())
