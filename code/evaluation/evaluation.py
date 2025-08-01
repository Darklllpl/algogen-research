import json
from pathlib import Path
import argparse
from tqdm import tqdm

def classify_error(gt_delta, pred_delta):
    """
    Heuristic for classifying the first mismatched delta.
    """
    gt_ops = gt_delta.get("operations", [])
    pred_ops = pred_delta.get("operations", [])
    gt_meta = gt_delta.get("meta", {})
    pred_meta = pred_delta.get("meta", {})

    # State update error
    if gt_meta != pred_meta:
        return "State Update Error"

    # Logical judgment error: prediction swaps, ground truth does not
    gt_has_swap = any(op.get("op") == "moveElements" for group in gt_ops for op in (group if isinstance(group, list) else [group]))
    pred_has_swap = any(op.get("op") == "moveElements" for group in pred_ops for op in (group if isinstance(group, list) else [group]))
    if pred_has_swap and not gt_has_swap:
        return "Logical Judgment Error"

    # Operation selection error: compare first operation name
    try:
        gt_op_name = gt_ops[0][0]['op'] if isinstance(gt_ops[0], list) else gt_ops[0]['op']
        pred_op_name = pred_ops[0][0]['op'] if isinstance(pred_ops[0], list) else pred_ops[0]['op']
        if gt_op_name != pred_op_name:
            return "Operation Selection Error"
    except (IndexError, TypeError):
        pass

    # Default: logical error
    return "Logical Judgment Error"

def compare_svl_files(gt_path, pred_path):
    """
    Compare a ground truth SVL file and a predicted SVL file.
    Return dict with FFI and error type.
    """
    try:
        gt_data = json.loads(Path(gt_path).read_text(encoding='utf-8'))
        pred_data = json.loads(Path(pred_path).read_text(encoding='utf-8'))
    except (json.JSONDecodeError, FileNotFoundError):
        return {"error": "File Load or JSON Parse Error"}

    gt_deltas = gt_data.get("deltas", [])
    pred_deltas = pred_data.get("deltas", [])

    # Find first frame mismatch
    for i in range(min(len(gt_deltas), len(pred_deltas))):
        gt_delta_str = json.dumps(gt_deltas[i], sort_keys=True)
        pred_delta_str = json.dumps(pred_deltas[i], sort_keys=True)
        if gt_delta_str != pred_delta_str:
            error_type = classify_error(gt_deltas[i], pred_deltas[i])
            return {
                "ffi": i + 1, 
                "error_type": error_type,
                "ground_truth_delta": gt_deltas[i],
                "prediction_delta": pred_deltas[i]
            }

    # Prediction ends early
    if len(pred_deltas) < len(gt_deltas):
        return {"ffi": len(pred_deltas) + 1, "error_type": "Premature Termination"}

    # All matched
    return {"ffi": len(gt_deltas), "error_type": None}

def main(args):
    gt_dir = Path(args.ground_truth_dir)
    pred_dir = Path(args.prediction_dir)
    stats = {
        "total_files": 0,
        "parse_errors": 0,
        "ffi_scores": [],
        "error_counts": {}
    }
    detailed_results = []

    gt_files = {p.name: p for p in gt_dir.rglob("*.json")}
    pred_files = {p.name: p for p in pred_dir.rglob("*.json")}
    common_files = gt_files.keys() & pred_files.keys()

    print(f"Found {len(common_files)} files to evaluate...")

    for filename in tqdm(common_files, desc="Evaluating files"):
        stats["total_files"] += 1
        gt_path = gt_files[filename]
        pred_path = pred_files[filename]
        result = compare_svl_files(gt_path, pred_path)
        if "error" in result:
            stats["parse_errors"] += 1
            error_type = "Syntactic Error"
            ffi_val = None
            gt_delta = None
            pred_delta = None
        else:
            ffi_val = result["ffi"]
            stats["ffi_scores"].append(ffi_val)
            error_type = result["error_type"]
            gt_delta = result.get("ground_truth_delta")
            pred_delta = result.get("prediction_delta")
        # Error statistics
        if error_type:
            stats["error_counts"][error_type] = stats["error_counts"].get(error_type, 0) + 1
        detailed_results.append({
            "filename": filename,
            "ffi": ffi_val,
            "error_type": error_type,
            "ground_truth_delta": gt_delta,
            "prediction_delta": pred_delta
        })

    # --- Print report ---
    print("\n--- Evaluation Report ---")
    print(f"Total files: {stats['total_files']}")
    print(f"Parse errors: {stats['parse_errors']}")
    if stats["ffi_scores"]:
        avg_ffi = sum(stats["ffi_scores"]) / len(stats["ffi_scores"])
        print(f"\nAverage FFI: {avg_ffi:.2f}")
        print("(How many steps model is correct on average)")
    if stats["error_counts"]:
        print("\nError type statistics:")
        total_errors = sum(stats["error_counts"].values())
        for error, count in sorted(stats["error_counts"].items(), key=lambda item: item[1], reverse=True):
            percentage = (count / total_errors) * 100 if total_errors > 0 else 0
            print(f"  - {error:<25}: {count} ({percentage:.1f}%)")
    print("\n--- End of Report ---")
    # Save results
    output_dir = Path("evaluation_results/")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_files": stats["total_files"],
        "parse_errors": stats["parse_errors"],
        "average_ffi": avg_ffi if stats["ffi_scores"] else None,
        "error_counts": stats["error_counts"]
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    with open(output_dir / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=4)
    print(f"Saved results to {output_dir / 'summary.json'} and {output_dir / 'detailed_results.json'}")

    output_report = {
        "summary": {
            "total_files": stats["total_files"],
            "parse_errors": stats["parse_errors"],
            "average_ffi": avg_ffi if stats["ffi_scores"] else None,
            "median_ffi": sorted(stats["ffi_scores"])[len(stats["ffi_scores"])//2] if stats["ffi_scores"] else None
        },
        "error_counts": stats["error_counts"],
        "raw_ffi_scores": stats["ffi_scores"]
    }
    report_filename = pred_dir.name + "_evaluation_report.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(output_report, f, indent=2)
    print(f"\nDetailed evaluation report saved to: {report_filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare and evaluate model-generated SVL files vs ground-truth SVL files.")
    parser.add_argument("ground_truth_dir", type=str, help="Directory with ground-truth SVL JSON files")
    parser.add_argument("prediction_dir", type=str, help="Directory with model-generated SVL JSON files")
    args = parser.parse_args()
    main(args)
