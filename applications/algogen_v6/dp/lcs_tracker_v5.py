import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_lcs_svl_v5(str1: str, str2: str):
    """
    Generate complete visualization sequence for Longest Common Subsequence (LCS) algorithm according to SVL 5.0 specification.
    This version uses auxiliary_views to clearly display DP table, input strings, and state transition dependencies.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Longest Common Subsequence (LCS)",
        "family": "Dynamic Programming"
    }

    variables_schema = [
        {"name": "i", "type": "pointer", "description": "Pointer for str1"},
        {"name": "j", "type": "pointer", "description": "Pointer for str2"},
        {"name": "LCS", "type": "value", "description": "Final longest common subsequence found"}
    ]

    data_schema = {} # LCS core data is in auxiliary views, main data structure is empty
    
    pseudocode = [
        "function LCS(str1, str2):",
        "  m = len(str1), n = len(str2)",
        "  dp = new table[m+1][n+1]",
        "  for i from 1 to m:",
        "    for j from 1 to n:",
        "      if str1[i-1] == str2[j-1]:",
        "        dp[i][j] = 1 + dp[i-1][j-1]",
        "      else:",
        "        dp[i][j] = max(dp[i-1][j], dp[i][j-1])",
        "  // Backtrack from dp[m][n] to find LCS",
        "  return LCS_string"
    ]

    # 2. Build initial frame
    m, n = len(str1), len(str2)
    
    # Initialize DP table, all values are 0
    dp_table_data = [[0] * (n + 1) for _ in range(m + 1)]

    # Create auxiliary views
    aux_views = [
        {
            "view_id": "input_strings", "type": "list", "title": "Input Strings",
            "data": {"String 1": list(str1), "String 2": list(str2)}
        },
        {
            "view_id": "dp_table", "type": "table", "title": "DP Table",
            "data": dp_table_data,
            "options": {
                "row_headers": ["-"] + list(str1),
                "col_headers": ["-"] + list(str2)
            }
        }
    ]
    
    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "array", "data": []}, # Main data is empty, all content is in auxiliary views
        "auxiliary_views": aux_views,
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    dp = [[0] * (n + 1) for _ in range(m + 1)] # Maintain DP table in memory
    
    deltas.append({"meta": {}, "code_highlight": 4, "operations": []})
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            meta = {"i": i, "j": j}
            
            # Highlight current cell being calculated and corresponding characters
            deltas.append({"meta": meta, "code_highlight": 5, "operations": [
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "current_cell"}}
            ]})
            
            deltas.append({"meta": meta, "code_highlight": 6, "operations": []})
            if str1[i-1] == str2[j-1]:
                # Characters match, depend on upper-left cell
                dp[i][j] = 1 + dp[i-1][j-1]
                deltas.append({"meta": meta, "code_highlight": 7, "operations": [
                    {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": [{"row": i-1, "col": j-1}], "to_cell": {"row": i, "col": j}, "styleKey": "dependency_arrow"}},
                    {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i-1, "col": j-1}], "styleKey": "dependency_cell"}}
                ]})
            else:
                # Characters don't match, depend on upper or left cell with larger value
                from_cells = []
                if dp[i-1][j] >= dp[i][j-1]:
                    dp[i][j] = dp[i-1][j]
                    from_cells.append({"row": i-1, "col": j})
                else:
                    dp[i][j] = dp[i][j-1]
                    from_cells.append({"row": i, "col": j-1})

                deltas.append({"meta": meta, "code_highlight": 9, "operations": [
                    {"op": "showDependency", "params": {"view_id": "dp_table", "from_cells": from_cells, "to_cell": {"row": i, "col": j}, "styleKey": "dependency_arrow"}},
                    {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": from_cells, "styleKey": "dependency_cell"}}
                ]})
            
            # Update cell value and highlight
            deltas.append({"meta": meta, "code_highlight": 7, "operations": [
                {"op": "updateTableCell", "params": {"view_id": "dp_table", "updates": [{"row": i, "col": j, "value": dp[i][j]}]}},
                {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "updated_cell"}}
            ]})

    # Backtracking phase
    lcs_str = ""
    i, j = m, n
    deltas.append({"meta": {"i": i, "j": j}, "code_highlight": 10, "operations": [
        {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "key_element"}}
    ]})
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs_str = str1[i-1] + lcs_str
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
        
        deltas.append({"meta": {"i": i, "j": j, "LCS": lcs_str}, "code_highlight": 10, "operations": [
            {"op": "highlightTableCell", "params": {"view_id": "dp_table", "cells": [{"row": i, "col": j}], "styleKey": "key_element"}}
        ]})

    deltas.append({"meta": {"LCS": lcs_str}, "code_highlight": 11, "operations": []})

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
    string1 = "AGGTAB"
    string2 = "GXTXAYB"
    
    output_filename = "lcs_svl_5.0.json"
    output_dir = Path("json_v5/dp") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating LCS SVL 5.0 sequence for strings '{string1}' and '{string2}'...")
    svl_output = generate_lcs_svl_v5(string1, string2)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (LCS) sequence saved to: {output_path}")