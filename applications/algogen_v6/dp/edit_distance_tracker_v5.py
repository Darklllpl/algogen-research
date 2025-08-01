import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_edit_distance_svl_v5(str1: str, str2: str):
    """
    Generate complete visualization sequence for Edit Distance algorithm according to SVL 5.0 specification.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Edit Distance (Levenshtein)",
        "family": "Dynamic Programming"
    }

    variables_schema = [
        {"name": "i", "type": "pointer", "description": "str1 pointer"},
        {"name": "j", "type": "pointer", "description": "str2 pointer"},
        {"name": "cost", "type": "value", "description": "substitution cost (0 or 1)"}
    ]

    data_schema = {}
    
    pseudocode = [
        "function EditDistance(str1, str2):",
        "  m = len(str1), n = len(str2)",
        "  dp = new table[m+1][n+1]",
        "  // Initialize base cases",
        "  for i from 0 to m: dp[i][0] = i",
        "  for j from 0 to n: dp[0][j] = j",
        "  // Fill DP table",
        "  for i from 1 to m:",
        "    for j from 1 to n:",
        "      cost = (str1[i-1] == str2[j-1]) ? 0 : 1",
        "      dp[i][j] = min(dp[i-1][j] + 1,          // Deletion",
        "                     dp[i][j-1] + 1,          // Insertion",
        "                     dp[i-1][j-1] + cost)     // Substitution/Match",
        "  return dp[m][n]"
    ]

    # 2. Build initial frame
    m, n = len(str1), len(str2)
    
    # Initialize DP table with base values
    dp_table_data = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp_table_data[i][0] = i
    for j in range(n + 1):
        dp_table_data[0][j] = j

    # Create auxiliary views
    aux_views = [
        {
            "view_id": "input_strings", "type": "list", "title": "Input Strings",
            "data": {"String 1 (s1)": list(str1), "String 2 (s2)": list(str2)}
        },
        {
            "view_id": "dp_table", "type": "table", "title": "DP Table (Edit Distance)",
            "data": dp_table_data,
            "options": {
                "row_headers": ["-"] + list(str1),
                "col_headers": ["-"] + list(str2)
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
    dp = [[0] * (n + 1) for _ in range(m + 1)] # Maintain DP table in memory
    
    # Initialize base cases
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    deltas.append({"meta": {}, "code_highlight": 5, "operations": [
        {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": r, "col": 0} for r in range(m+1)], "styleKey": "updated_cell"}}
    ]})
    deltas.append({"meta": {}, "code_highlight": 6, "operations": [
        {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": 0, "col": c} for c in range(n+1)], "styleKey": "updated_cell"}}
    ]})
    
    deltas.append({"meta": {}, "code_highlight": 8, "operations": []})
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            meta = {"i": i, "j": j}
            
            # Highlight current cell and corresponding characters
            deltas.append({"meta": meta, "code_highlight": 9, "operations": [
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "current_cell"}}
            ]})
            
            # Show dependencies from three directions
            from_cells = [{"row": i-1, "col": j}, {"row": i, "col": j-1}, {"row": i-1, "col": j-1}]
            deltas.append({"meta": meta, "code_highlight": 11, "operations": [
                {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": from_cells, "to_cell": {"row": i, "col": j}, "styleKey": "dependency_arrow"}},
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": from_cells, "styleKey": "dependency_cell"}}
            ]})
            
            cost = 0 if str1[i-1] == str2[j-1] else 1
            meta["cost"] = cost
            deltas.append({"meta": meta, "code_highlight": 10, "operations": []})

            # Calculate new value
            dp[i][j] = min(dp[i-1][j] + 1,          # Deletion
                           dp[i][j-1] + 1,          # Insertion
                           dp[i-1][j-1] + cost)     # Substitution / Match
            
            # Update cell value and highlight
            deltas.append({"meta": meta, "code_highlight": 11, "operations": [
                {"op": "updateTableCell", "params": {"view_id": "dp_table", "updates": [{"row": i, "col": j, "value": dp[i][j]}]}},
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "updated_cell"}}
            ]})

    # Final result
    final_distance = dp[m][n]
    deltas.append({"meta": {"i": m, "j": n}, "code_highlight": 14, "operations": [
        {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": m, "col": n}], "styleKey": "key_element"}}
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
    string1 = "saturday"
    string2 = "sunday"
    
    output_filename = "edit_distance_svl_5.0.json"
    output_dir = Path("json_v5/dp") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating Edit Distance SVL 5.0 sequence for strings '{string1}' and '{string2}'...")
    svl_output = generate_edit_distance_svl_v5(string1, string2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (Edit Distance) sequence saved to: {output_path}")