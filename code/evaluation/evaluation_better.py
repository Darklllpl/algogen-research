import json
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def classify_error(gt_delta, pred_delta):
    """
    Heuristic to classify the first wrong delta.
    """
    gt_ops = gt_delta.get("operations", [])
    pred_ops = pred_delta.get("operations", [])
    gt_meta = gt_delta.get("meta", {})
    pred_meta = pred_delta.get("meta", {})

    # State update error
    if gt_meta != pred_meta:
        return "State Update Error"

    # Logical judgment error
    gt_has_swap = any(op.get("op") == "moveElements" for group in gt_ops for op in (group if isinstance(group, list) else [group]))
    pred_has_swap = any(op.get("op") == "moveElements" for group in pred_ops for op in (group if isinstance(group, list) else [group]))
    if pred_has_swap and not gt_has_swap:
        return "Logical Judgment Error"

    # Operation selection error
    try:
        gt_op_name = gt_ops[0][0]['op'] if isinstance(gt_ops[0], list) else gt_ops[0]['op']
        pred_op_name = pred_ops[0][0]['op'] if isinstance(pred_ops[0], list) else pred_ops[0]['op']
        if gt_op_name != pred_op_name:
            return "Operation Selection Error"
    except:
        pass

    return "Logical Judgment Error"

def compare_svl_files(gt_path, pred_path):
    """
    Compare one ground-truth SVL file with a predicted SVL file.
    Returns FFI and error type.
    """
    try:
        gt_data = json.loads(Path(gt_path).read_text(encoding='utf-8'))
        pred_data = json.loads(Path(pred_path).read_text(encoding='utf-8'))
    except Exception:
        return {"error": "File Load or JSON Parse Error"}

    gt_deltas = gt_data.get("deltas", [])
    pred_deltas = pred_data.get("deltas", [])
    array_length = len(gt_data.get("initial_frame", {}).get("data_state", []))
    algo_name = gt_data.get("algorithm", {}).get("name", "Unknown")

    correct_gt = correct_pred = error_gt = error_pred = None
    min_len = min(len(gt_deltas), len(pred_deltas))
    mismatch_index = None
    for i in range(min_len):
        if json.dumps(gt_deltas[i], sort_keys=True) != json.dumps(pred_deltas[i], sort_keys=True):
            mismatch_index = i
            break

    if mismatch_index is not None:
        error_gt = gt_deltas[mismatch_index]
        error_pred = pred_deltas[mismatch_index]
        error_type = classify_error(error_gt, error_pred)
        ffi_val = mismatch_index + 1
        if mismatch_index > 0:
            correct_gt = gt_deltas[mismatch_index - 1]
            correct_pred = pred_deltas[mismatch_index - 1]
        return {
            "ffi": ffi_val,
            "error_type": error_type,
            "array_length": array_length,
            "algorithm": algo_name,
            "correct_ground_truth_delta": correct_gt,
            "correct_prediction_delta": correct_pred,
            "error_ground_truth_delta": error_gt,
            "error_prediction_delta": error_pred
        }

    if len(pred_deltas) < len(gt_deltas):
        ffi_val = len(pred_deltas) + 1
        error_type = "Premature Termination"
        if pred_deltas:
            correct_gt = gt_deltas[len(pred_deltas) - 1]
            correct_pred = pred_deltas[-1]
        error_gt = gt_deltas[len(pred_deltas)]
        return {
            "ffi": ffi_val,
            "error_type": error_type,
            "array_length": array_length,
            "algorithm": algo_name,
            "correct_ground_truth_delta": correct_gt,
            "correct_prediction_delta": correct_pred,
            "error_ground_truth_delta": error_gt,
            "error_prediction_delta": None
        }

    ffi_val = len(gt_deltas)
    error_type = None
    if gt_deltas:
        correct_gt = gt_deltas[-1]
        correct_pred = pred_deltas[-1]
    return {
        "ffi": ffi_val,
        "error_type": error_type,
        "array_length": array_length,
        "algorithm": algo_name,
        "correct_ground_truth_delta": correct_gt,
        "correct_prediction_delta": correct_pred,
        "error_ground_truth_delta": None,
        "error_prediction_delta": None
    }

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
    for filename in tqdm(common_files, desc="Evaluating"):
        stats["total_files"] += 1
        result = compare_svl_files(gt_files[filename], pred_files[filename])
        if "error" in result:
            stats["parse_errors"] += 1
            ffi_val = None
            error_type = "Syntactic Error"
            array_length = None
            algorithm = "Unknown"
            input_type = "Unknown"
        else:
            ffi_val = result["ffi"]
            stats["ffi_scores"].append(ffi_val)
            error_type = result["error_type"]
            array_length = result.get("array_length")
            algorithm = result.get("algorithm")
            # Extract input type from filename
            if "all_same" in filename:
                input_type = "All Same"
            elif "nearly_reversed" in filename:
                input_type = "Nearly Reversed"
            elif "nearly_sorted" in filename:
                input_type = "Nearly Sorted"
            elif "random" in filename:
                input_type = "Random"
            elif "with_duplicates" in filename:
                input_type = "With Duplicates"
            else:
                input_type = "Unknown"

        if error_type:
            stats["error_counts"][error_type] = stats["error_counts"].get(error_type, 0) + 1
        detailed_results.append({
            "filename": filename,
            "ffi": ffi_val,
            "error_type": error_type,
            "array_length": array_length,
            "algorithm": algorithm,
            "input_type": input_type,
            "correct_ground_truth_delta": result.get("correct_ground_truth_delta"),
            "correct_prediction_delta": result.get("correct_prediction_delta"),
            "error_ground_truth_delta": result.get("error_ground_truth_delta"),
            "error_prediction_delta": result.get("error_prediction_delta")
        })

    ffi_arr = np.array(stats["ffi_scores"]) if stats["ffi_scores"] else np.array([])
    summary = {
        "total_files": stats["total_files"],
        "parse_errors": stats["parse_errors"],
        "count": int(len(ffi_arr)),
        "average": float(np.mean(ffi_arr)) if ffi_arr.size else None,
        "median": float(np.median(ffi_arr)) if ffi_arr.size else None,
        "std_dev": float(np.std(ffi_arr, ddof=1)) if ffi_arr.size > 1 else None,
        "q1": float(np.percentile(ffi_arr, 25)) if ffi_arr.size else None,
        "q3": float(np.percentile(ffi_arr, 75)) if ffi_arr.size else None,
        "error_counts": stats["error_counts"]
    }

    print("\n--- Evaluation Report ---")
    print(f"Total files: {summary['total_files']}")
    print(f"Parse errors: {summary['parse_errors']}")
    if summary['count']:
        print(f"Sample count: {summary['count']}")
        print(f"Mean FFI: {summary['average']:.2f}")
        print(f"Median: {summary['median']:.2f}")
        print(f"Std: {summary['std_dev']:.2f}")
        print(f"Q1: {summary['q1']:.2f}, Q3: {summary['q3']:.2f}")
    if summary['error_counts']:
        print("Error types:")
        for et, cnt in summary['error_counts'].items():
            pct = cnt / sum(summary['error_counts'].values()) * 100
            print(f"  - {et}: {cnt} ({pct:.1f}%)")

    out_dir = Path("evaluation_results").mkdir(parents=True, exist_ok=True)
    with open(Path("evaluation_results") / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(Path("evaluation_results") / "detailed_results.json", 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    # FFI histogram
    if ffi_arr.size:
        plt.figure()
        plt.hist(ffi_arr, bins='auto')
        plt.title('FFI Distribution Histogram')
        plt.xlabel('FFI Steps')
        plt.ylabel('Number of Samples')
        plt.savefig(Path("evaluation_results") / "ffi_histogram.png")
        print("Saved histogram to evaluation_results/ffi_histogram.png")

        # FFI boxplot
        plt.figure()
        plt.boxplot(ffi_arr)
        plt.title('FFI Distribution Box Plot')
        plt.ylabel('FFI Steps')
        plt.savefig(Path("evaluation_results") / "ffi_boxplot.png")
        print("Saved boxplot to evaluation_results/ffi_boxplot.png")

    # FFI vs. array length scatter plot
    array_lengths = [res["array_length"] for res in detailed_results if res["ffi"] is not None and "array_length" in res]
    ffi_scores = [res["ffi"] for res in detailed_results if res["ffi"] is not None and "array_length" in res]
    if array_lengths:
        plt.figure()
        plt.scatter(array_lengths, ffi_scores)
        plt.title('FFI vs. Input Array Length')
        plt.xlabel('Input Array Length')
        plt.ylabel('FFI')
        plt.savefig(Path("evaluation_results") / "ffi_vs_array_length.png")
        print("Saved scatter plot to evaluation_results/ffi_vs_array_length.png")

    # FFI by algorithm bar chart
    algo_ffi = defaultdict(list)
    for res in detailed_results:
        if res["ffi"] is not None:
            algo_ffi[res["algorithm"]].append(res["ffi"])

    algorithms = list(algo_ffi.keys())
    avg_ffis = [np.mean(ffi_list) for ffi_list in algo_ffi.values()]

    if algorithms:
        plt.figure()
        plt.bar(algorithms, avg_ffis)
        plt.title('Average FFI by Algorithm Type')
        plt.xlabel('Algorithm')
        plt.ylabel('Average FFI')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path("evaluation_results") / "avg_ffi_by_algorithm.png")
        print("Saved bar chart to evaluation_results/avg_ffi_by_algorithm.png")

    # FFI by input type bar chart
    input_type_ffi = defaultdict(list)
    for res in detailed_results:
        if res["ffi"] is not None:
            input_type_ffi[res["input_type"]].append(res["ffi"])

    input_types = list(input_type_ffi.keys())
    avg_ffis = [np.mean(ffi_list) for ffi_list in input_type_ffi.values()]

    if input_types:
        plt.figure()
        plt.bar(input_types, avg_ffis)
        plt.title('Average FFI by Input Type')
        plt.xlabel('Input Type')
        plt.ylabel('Average FFI')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(Path("evaluation_results") / "avg_ffi_by_input_type.png")
        print("Saved bar chart to evaluation_results/avg_ffi_by_input_type.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare and evaluate SVL prediction results with rich analysis.")
    parser.add_argument('ground_truth_dir', type=str, help='Ground truth directory')
    parser.add_argument('prediction_dir', type=str, help='Prediction directory')
    args = parser.parse_args()
    main(args)
