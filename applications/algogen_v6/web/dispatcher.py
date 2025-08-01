# dispatcher.py
import json
from pathlib import Path
import sys
import argparse

sys.path.append(str(Path(__file__).parent.parent / 'sort'))
sys.path.append(str(Path(__file__).parent.parent / 'dp'))
sys.path.append(str(Path(__file__).parent.parent / 'graph'))
sys.path.append(str(Path(__file__).parent.parent))
from style_merger import merge_styles
from default_styles import DEFAULT_STYLES

# --- 1. Import main generation functions from various tracker files ---
# Assume all tracker files are in the same directory or in Python path
from bubble_sort_tracker_v5 import generate_bubble_sort_svl_v5
from insertion_sort_tracker_v5 import generate_insertion_sort_svl_v5
from selection_sort_tracker_v5 import generate_selection_sort_svl_v5
from quicksort_tracker_v5 import generate_quicksort_svl_v5
from merge_sort_tracker_v5 import generate_merge_sort_svl_v5
from heap_sort_tracker_v5 import generate_heap_sort_svl_v5
# ... Future imports for graph and DP tracker functions will be here ...
from bfs_tracker_v5 import generate_bfs_svl_v5
from dfs_tracker_v5 import generate_dfs_svl_v5
from dijkstra_tracker_v5 import generate_dijkstra_svl_v5
from bellman_ford_tracker_v5 import generate_bellman_ford_svl_v5
from prim_tracker_v5 import generate_prim_svl_v5
from kruskal_tracker_v5 import generate_kruskal_svl_v5
from topological_sort_tracker_v5 import generate_topological_sort_svl_v5

from lcs_tracker_v5 import generate_lcs_svl_v5
from knapsack_01_tracker_v5 import generate_knapsack_01_svl_v5
from edit_distance_tracker_v5 import generate_edit_distance_svl_v5

# --- 2. Build algorithm mapping table ---
# Map algorithm ID strings to actual Python function objects
ALGORITHM_DISPATCH_TABLE = {
    # Sorting algorithms
    "bubble_sort": generate_bubble_sort_svl_v5,
    "insertion_sort": generate_insertion_sort_svl_v5,
    "selection_sort": generate_selection_sort_svl_v5,
    "quick_sort": generate_quicksort_svl_v5,
    "merge_sort": generate_merge_sort_svl_v5,
    "heap_sort": generate_heap_sort_svl_v5,
    
    # Graph algorithms
    "bfs": generate_bfs_svl_v5,
    "dfs": generate_dfs_svl_v5,
    "dijkstra": generate_dijkstra_svl_v5,
    "bellman_ford": generate_bellman_ford_svl_v5,
    "prim": generate_prim_svl_v5,
    "kruskal": generate_kruskal_svl_v5,
    "topological_sort": generate_topological_sort_svl_v5,
    
    # DP algorithms
    "lcs": generate_lcs_svl_v5,
    "knapsack_01": generate_knapsack_01_svl_v5,
    "edit_distance": generate_edit_distance_svl_v5,
}

def dispatch_and_generate(intent_json: dict):
    """
    Main dispatch function. Receives LLM-generated intent JSON, calls corresponding tracker, and returns final SVL 5.0 object.
    """
    algorithm_id = intent_json.get("algorithm_id")
    data_input = intent_json.get("data_input")

    print(f"Dispatcher: Received request with algorithm ID '{algorithm_id}'.")

    # --- 3. Find corresponding tracker function from mapping table ---
    tracker_function = ALGORITHM_DISPATCH_TABLE.get(algorithm_id)

    if not tracker_function:
        print(f"Error: Algorithm with ID '{algorithm_id}' not found in dispatch table.")
        return None
    
    if data_input is None:
        print(f"Error: Missing 'data_input' in intent JSON.")
        return None

    # --- 4. Call function and pass parameters ---
    try:
        print(f"Calling {tracker_function.__name__} with data: {data_input}")
        
        param_extractors = {
            "bubble_sort": lambda d: (d,),
            "insertion_sort": lambda d: (d,),
            "selection_sort": lambda d: (d,),
            "quick_sort": lambda d: (d,),
            "merge_sort": lambda d: (d,),
            "heap_sort": lambda d: (d,),
            "bfs": lambda d: (d["graph"], d.get("start_node", None)),
            "dfs": lambda d: (d["graph"], d.get("start_node", None)),
            "dijkstra": lambda d: (d["graph"], d.get("start_node", None)),
            "bellman_ford": lambda d: (d["graph"], d.get("start_node", None)),
            "prim": lambda d: (d["graph"], d.get("start_node", None)),
            "kruskal": lambda d: (d["graph"],),
            "topological_sort": lambda d: (d["graph"],),
            "lcs": lambda d: (d["str1"], d["str2"]),
            "knapsack_01": lambda d: (d["items"], d["capacity"]),
            "edit_distance": lambda d: (d["str1"], d["str2"]),
        }
        
        if algorithm_id in param_extractors:
            args = param_extractors[algorithm_id](data_input)
            svl_object = tracker_function(*args)
        else:
            svl_object = tracker_function(data_input)

        print("Tracker successfully generated SVL 5.0 object.")
        return svl_object

    except Exception as e:
        print(f"Error calling tracker '{tracker_function.__name__}': {e}")
        return None

# --- Usage example ---
if __name__ == '__main__':
    # Use argparse to support loading intent JSON from external file
    parser = argparse.ArgumentParser(description='Dispatch based on intent JSON')
    parser.add_argument('intent_file', nargs='?', help='Path to intent JSON file')
    args = parser.parse_args()
    if args.intent_file:
        with open(args.intent_file, 'r', encoding='utf-8') as f:
            sample_intent = json.load(f)
    else:
        # Simulate LLM-generated intent JSON
        sample_intent = {
            "algorithm_id": "bubble_sort",
            "data_input": [5, 1, 4],
            "style_overrides": {
                "elementStyles": {
                    "idle": {"fill": "#E8F5E9"},
                    "compare": {"fill": "#F8BBD0"},
                    "sorted": {"fill": "#C8E6C9"},
                    "swapping": {"fill": "#F48FB1"},
                    "pivot": {"fill": "#FFF9C4"},
                    "key_element": {"fill": "#A5D6A7"},
                    "shifting": {"fill": "#B2DFDB"},
                    "sub_array_active": {"fill": "#FCE4EC"},
                    "partition_area": {"fill": "#E8F5E9"},
                    "in_path_node": {"fill": "#C8E6C9"},
                    "updated_cell": {"fill": "rgba(200, 230, 201, 0.7)"}
                }
            }
        }
    # Merge styles: get override styles from sample_intent and generate final_styles.py
    style_overrides = sample_intent.get("style_overrides", {})
    final_styles = merge_styles(DEFAULT_STYLES, style_overrides)
    with open('final_styles.py', 'w', encoding='utf-8') as f:
        f.write('# final_styles.py\n')
        f.write('# Generated by dispatcher.py\n\n')
        f.write('DEFAULT_STYLES = ')
        f.write(json.dumps(final_styles, indent=2, ensure_ascii=False))
        f.write('\n')
    # Call dispatcher
    final_svl = dispatch_and_generate(sample_intent)

    if final_svl:
        # Save generated SVL object to file, output directory is in same directory as dispatcher.py
        output_dir = Path(__file__).parent / "dispatch_output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{sample_intent['algorithm_id']}_svl_5.0.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_svl, f, indent=2, ensure_ascii=False)
        
        print(f"Dispatch successful! Final SVL file saved to: {output_path}")