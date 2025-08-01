import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_knapsack_01_svl_v5(items: list, capacity: int):
    """
    Generate complete visualization sequence for 0/1 Knapsack problem according to SVL 5.0 specification.
    This version uses two auxiliary views to clearly display item information and DP table state transitions.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "0/1 Knapsack Problem",
        "family": "Dynamic Programming"
    }

    variables_schema = [
        {"name": "i", "type": "pointer", "description": "Current item being considered"},
        {"name": "w", "type": "pointer", "description": "Current knapsack capacity"},
        {"name": "maxValue", "type": "value", "description": "Final maximum value obtained"}
    ]

    data_schema = {}
    
    pseudocode = [
        "function Knapsack(items, W):",
        "  n = len(items)",
        "  dp = new table[n+1][W+1]",
        "  for i from 1 to n:",
        "    for w from 1 to W:",
        "      if items[i-1].weight <= w:",
        "        dp[i][w] = max(dp[i-1][w], items[i-1].value + dp[i-1][w-items[i-1].weight])",
        "      else:",
        "        dp[i][w] = dp[i-1][w]",
        "  return dp[n][W]"
    ]

    # 2. Build initial frame
    num_items = len(items)
    
    # Initialize DP table, all values are 0
    dp_table_data = [[0] * (capacity + 1) for _ in range(num_items + 1)]
    
    # Create auxiliary views
    items_table_data = [[item['weight'], item['value']] for item in items]

    aux_views = [
        {
            "view_id": "items_table", "type": "table", "title": "Items",
            "data": items_table_data,
            "options": {
                "row_headers": [f"Item {i+1}" for i in range(num_items)],
                "col_headers": ["Weight", "Value"]
            }
        },
        {
            "view_id": "dp_table", "type": "table", "title": "DP Table (Max Value)",
            "data": dp_table_data,
            "options": {
                "row_headers": [f"i={i}" for i in range(num_items + 1)],
                "col_headers": [f"w={w}" for w in range(capacity + 1)]
            }
        }
    ]
    
    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "array", "data": []},
        "auxiliary_views": aux_views,
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    dp = [[0] * (capacity + 1) for _ in range(num_items + 1)] # Maintain DP table in memory
    
    deltas.append({"meta": {}, "code_highlight": 4, "operations": []})
    for i in range(1, num_items + 1):
        item_weight = items[i-1]['weight']
        item_value = items[i-1]['value']
        
        deltas.append({"meta": {"i": i}, "code_highlight": 5, "operations": [
            {"op": "highlightTableCell", "params": {"view_id": "items_table", "cells": [{"row": i-1, "col": 0}, {"row": i-1, "col": 1}], "styleKey": "key_element"}}
        ]})

        for w in range(1, capacity + 1):
            meta = {"i": i, "w": w}
            
            # Highlight current DP cell being calculated
            deltas.append({"meta": meta, "code_highlight": 5, "operations": [
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": w}], "styleKey": "current_cell"}}
            ]})
            
            deltas.append({"meta": meta, "code_highlight": 6, "operations": []})
            if item_weight <= w:
                # Item can be included, show dependencies from two decisions
                dp[i][w] = max(dp[i-1][w], item_value + dp[i-1][w - item_weight])
                deltas.append({"meta": meta, "code_highlight": 7, "operations": [
                    {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": [{"row": i-1, "col": w}], "to_cell": {"row": i, "col": w}, "styleKey": "dependency_arrow"}},
                    {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": [{"row": i-1, "col": w - item_weight}], "to_cell": {"row": i, "col": w}, "styleKey": "dependency_arrow"}},
                    {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i-1, "col": w}, {"row": i-1, "col": w - item_weight}], "styleKey": "dependency_cell"}}
                ]})
            else:
                # Item cannot be included, depend on cell above
                dp[i][w] = dp[i-1][w]
                deltas.append({"meta": meta, "code_highlight": 9, "operations": [
                     {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": [{"row": i-1, "col": w}], "to_cell": {"row": i, "col": w}, "styleKey": "dependency_arrow"}},
                     {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i-1, "col": w}], "styleKey": "dependency_cell"}}
                ]})
            
            # Update cell value
            deltas.append({"meta": meta, "code_highlight": 7, "operations": [
                {"op": "updateTableCell", "params": {"view_id": "dp_table", "updates": [{"row": i, "col": w, "value": dp[i][w]}]}},
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": w}], "styleKey": "updated_cell"}}
            ]})
        
        # Restore item row highlight
        deltas.append({"meta": {"i": i}, "code_highlight": 4, "operations": [
            {"op": "highlightTableCell", "params": {"view_id": "items_table", "cells": [{"row": i-1, "col": 0}, {"row": i-1, "col": 1}], "styleKey": "idle"}}
        ]})

    # Backtrack to find optimal solution
    max_value = dp[num_items][capacity]
    w = capacity
    selected_items_indices = []
    for i in range(num_items, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected_items_indices.append(i-1)
            w -= items[i-1]['weight']
    
    deltas.append({"meta": {"maxValue": max_value}, "code_highlight": 10, "operations": [
        {"op": "highlightTableCell", "params": {"view_id": "items_table", "cells": [{"row": r, "col": c} for r in selected_items_indices for c in range(2)], "styleKey": "in_path_node"}}
    ]})
    
    # 4. Assemble and return final SVL object
    final_svl_object = {
        "svl_version": "5.0",
        "algorithm": algorithm_info,
        "initial_frame": initial_frame,
        "deltas": deltas
    }
    
    return final_svl_object

# Usage example
if __name__ == '__main__':
    # Items list: [(weight, value), ...]
    items_data = [{'weight': 1, 'value': 1}, {'weight': 2, 'value': 6}, {'weight': 5, 'value': 18}, {'weight': 6, 'value': 22}, {'weight': 7, 'value': 28}]
    knapsack_capacity = 11
    
    output_filename = "knapsack_01_svl_5.0.json"
    output_dir = Path("json_v5/dp") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating 0/1 Knapsack SVL 5.0 sequence (capacity: {knapsack_capacity})...")
    svl_output = generate_knapsack_01_svl_v5(items_data, knapsack_capacity)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (0/1 Knapsack) sequence saved to: {output_path}")