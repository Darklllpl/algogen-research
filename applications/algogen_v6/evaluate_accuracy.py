import json
from pathlib import Path
from typing import Any
import unicodedata


def normalize_text(text):
    """Normalize string with NFKC and remove leading/trailing spaces"""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKC", text).strip()


def normalize_json(obj: Any) -> Any:
    """
    Normalize JSON structure:
    - Remove empty dict, empty list and None
    - Recursively clean nested structures
    - Apply Unicode normalization and strip to all strings
    """
    if isinstance(obj, dict):
        return {
            k: normalize_json(v)
            for k, v in obj.items()
            if v not in ({}, [], None)
        }
    elif isinstance(obj, list):
        return [normalize_json(item) for item in obj if item not in ({}, [], None)]
    elif isinstance(obj, str):
        return normalize_text(obj)
    else:
        return obj


def adjacency_lists_equivalent(g1: Any, g2: Any) -> bool:
    """
    Compare if two adjacency lists (dict of list of [node, weight]) are equivalent,
    ignoring node and edge order.
    """
    if not isinstance(g1, dict) or not isinstance(g2, dict):
        return False
    if set(g1.keys()) != set(g2.keys()):
        return False

    def norm_adj(adj):
        result = {}
        for node, nbrs in adj.items():
            tuples = [tuple(n) for n in nbrs]
            result[node] = sorted(tuples)
        return {k: result[k] for k in sorted(result)}

    return norm_adj(g1) == norm_adj(g2)


def is_equivalent(gold: Any, model: Any) -> bool:
    """
    Check if gold is covered by model (lenient mode):
    - Each key-value pair in gold exists in model
    - Ignore case differences, field order, redundant keys
    """
    gold = normalize_json(gold)
    model = normalize_json(model)

    if isinstance(gold, dict) and isinstance(model, dict):
        if all(isinstance(v, list) and all(isinstance(e, list) and len(e) == 2 for e in v)
               for v in gold.values()):
            return adjacency_lists_equivalent(gold, model)

    if isinstance(gold, dict):
        if not isinstance(model, dict):
            return False
        for key, gold_value in gold.items():
            if key not in model:
                return False
            if not is_equivalent(gold_value, model[key]):
                return False
        return True

    elif isinstance(gold, list):
        if not isinstance(model, list):
            return False
        def match_one(item):
            return any(is_equivalent(item, m) for m in model)
        return all(match_one(item) for item in gold)

    elif isinstance(gold, str) and isinstance(model, str):
        return gold.lower() == model.lower()

    else:
        return gold == model


def is_equivalent_debug(gold: Any, model: Any, path: str = "<root>") -> (bool, str):
    """
    Lenient comparison and return failure reason:
    - Use unordered comparison for adjacency lists
    - Other structures follow is_equivalent logic recursively
    """
    gold_norm = normalize_json(gold)
    model_norm = normalize_json(model)

    if isinstance(gold_norm, dict) and isinstance(model_norm, dict):
        if path.endswith("data_input.graph"):
            if adjacency_lists_equivalent(gold_norm, model_norm):
                return True, ''
            else:
                return False, f"Field {path} adjacency list content not equivalent (edges or order different)"

    if isinstance(gold_norm, dict): 
        if not isinstance(model_norm, dict):
            return False, f"Field {path} type mismatch, gold is dict, model is {type(model_norm).__name__}"
        for key, gold_value in gold_norm.items():
            if key not in model_norm:
                return False, f"Field {path}.{key} missing in model"
            ok, reason = is_equivalent_debug(
                gold_value,
                model_norm[key],
                f"{path}.{key}"
            )
            if not ok:
                return False, reason
        return True, ''

    elif isinstance(gold_norm, list):
        if not isinstance(model_norm, list):
            return False, f"Field {path} type mismatch, gold is list, model is {type(model_norm).__name__}"
        for idx, gold_item in enumerate(gold_norm):
            found = False
            for model_item in model_norm:
                if is_equivalent(gold_item, model_item):
                    found = True
                    break
            if not found:
                return False, f"Field {path}[{idx}] not found equivalent item in model list"
        return True, ''

    elif isinstance(gold_norm, str) and isinstance(model_norm, str):
        if gold_norm.lower() != model_norm.lower():
            return False, f"Field {path} string mismatch, gold: '{gold_norm}', model: '{model_norm}'"
        return True, ''

    else:
        if gold_norm != model_norm:
            return False, f"Field {path} value mismatch, gold: {gold_norm}, model: {model_norm}"
        return True, ''


def evaluate_jsonl_relaxed(file_path: str):
    """
    Read .jsonl file, leniently evaluate each record, and output pass/fail statistics and reasons.
    """
    total = correct = 0
    file_path = Path(file_path)
    lines = file_path.read_text(encoding='utf-8').splitlines()

    for idx, line in enumerate(lines, 1):
        data = json.loads(line)
        gold = data.get('gold_standard_output', {})
        model = data.get('model_output')

        if model is None:
            print(f"Record {idx}: ❌No model_output, failed")
            continue

        mdi = model.get("data_input")
        if isinstance(mdi, list) and not mdi:
            print(f"Record {idx}: ❌model_output.data_input is empty, failed")
            continue

        ok, reason = is_equivalent_debug(gold, model, "<root>")
        total += 1
        if ok:
            print(f"Record {idx}: ✅Passed")
            correct += 1
        else:
            print(f"Record {idx}: ❌Failed, reason: {reason}")

    if total:
        print(f"\nTotal: {total}, Passed: {correct}, Pass rate: {correct/total:.2%}")
    else:
        print("\nNo data to evaluate")


if __name__ == '__main__':
    evaluate_jsonl_relaxed('wrong.jsonl')
