import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_bubble_sort_svl_v5(initial_array: list):
    """
    Generate a complete, compliant visualization sequence object for bubble sort according to final SVL 5.0 specification.
    This version strictly aligns with final SVL 5.0.md and svl5_schema.json.
    """
    
    # =================================================================
    # 1. Define static components (compliant with specification)
    # =================================================================
    
    # Corresponds to algorithm object, name and family are required fields.
    algorithm_info = {
        "name": "Bubble Sort",
        "family": "Sorting"
    }

    # Corresponds to variables_schema
    variables_schema = [
        {"name": "i", "type": "pointer", "description": "Outer loop, controls sorted region boundary"},
        {"name": "j", "type": "pointer", "description": "Inner loop, for comparing and swapping adjacent elements"}
    ]
    
    # Corresponds to pseudocode, line numbers start from 1
    pseudocode = [
        "function BubbleSort(array):",              # 1
        "  n = length(array)",                      # 2
        "  for i from 0 to n-2:",                   # 3
        "    for j from 0 to n-i-2:",               # 4
        "      if array[j] > array[j+1]:",         # 5
        "        swap(array[j], array[j+1])",       # 6
        "  return array"                            # 7
    ]

    # =================================================================
    # 2. Build initial frame (initial_frame)
    # =================================================================
    
    initial_array_state = [
        {"index": i, "value": val, "state": "idle"}
        for i, val in enumerate(initial_array)
    ]
    
    # Build initial_frame, including all Tier-1 required fields
    initial_frame = {
        "data_schema": {}, # For simple arrays, this can be empty object
        "data_state": {
            "type": "array",
            "data": initial_array_state
        },
        "variables_schema": variables_schema,
        "pseudocode": pseudocode,
        "code_highlight": 1,
        "styles": DEFAULT_STYLES # Directly use imported shared styles
    }

    # =================================================================
    # 3. Track algorithm execution, generate Deltas
    # =================================================================
    
    deltas = []
    arr = list(initial_array)
    n = len(arr)
    
    deltas.append({"meta": {}, "code_highlight": 2, "operations": []})
    deltas.append({"meta": {}, "code_highlight": 3, "operations": []})
    for i in range(n - 1):
        # Mark outer loop variable i
        deltas.append({"meta": {"i": i, "j": "-"}, "code_highlight": 4, "operations": []})
        
        for j in range(0, n - i - 1):
            current_meta = {"i": i, "j": j}
            
            # >> Delta: Highlight comparison of j and j+1 <<
            deltas.append({
                "meta": current_meta,
                "code_highlight": 5,
                "operations": [
                    [{"op": "updateStyle", "params": {"indices": [j, j + 1], "styleKey": "compare"}}]
                ]
            })

            # Comparison logic
            if arr[j] > arr[j + 1]:
                # >> Delta: Execute move <<
                # Actually update array
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                
                deltas.append({
                    "meta": current_meta,
                    "code_highlight": 6,
                    "operations": [
                        {"op": "moveElements", "params": {
                            "pairs": [{"fromIndex": j, "toIndex": j + 1}, {"fromIndex": j + 1, "toIndex": j}]
                        }},
                    ]
                })

            # >> Delta: 恢复比较过的元素的样式 <<
            # 无论是否交换，比较过的元素都恢复为 idle 状态，准备下一次比较
            deltas.append({
                "meta": current_meta,
                "code_highlight": 4, # 回到内层 for 循环的头部
                "operations": [
                    {"op": "updateStyle", "params": {"indices": [j, j + 1], "styleKey": "idle"}}
                ]
            })

        # >> Delta: 一轮外层循环结束，将该轮找到的最大值标记为已排序 <<
        sorted_index = n - 1 - i
        deltas.append({
            "meta": {"i": i, "j": "-"}, 
            "code_highlight": 3, # 回到外层 for 循环的头部
            "operations": [
                {"op": "updateStyle", "params": {"indices": [sorted_index], "styleKey": "sorted"}}
            ]
        })
            
    # >> Delta: 排序结束，将所有元素标记为已排序 <<
    # 最后一个元素(索引0)在循环结束后自动有序，但也需标记
    final_ops = [{"op": "updateStyle", "params": {"indices": list(range(n)), "styleKey": "sorted"}}]
    deltas.append({"meta": {}, "code_highlight": 7, "operations": final_ops})

    # =================================================================
    # 4. 组装并返回最终的SVL对象
    # =================================================================
    
    final_svl_object = {
        "svl_version": "5.0",
        "algorithm": algorithm_info,
        "initial_frame": initial_frame,
        "deltas": deltas
    }
    
    return final_svl_object


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 定义输入和输出
    my_array = [5, 1, 4, 2, 8]
    output_filename = "bubble_sort_svl_5.0.json"
    
    # 确保输出目录存在
    output_dir = Path("json_v5/sort") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    # 2. 运行追踪器生成完整的SVL对象
    print(f"正在为数组 {my_array} 生成冒泡排序的SVL 5.0序列...")
    svl_output = generate_bubble_sort_svl_v5(my_array)

    # 3. 将SVL对象写入JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        # 使用 ensure_ascii=False 保证中文字符正确显示
        # 使用 indent=2 保持可读性
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 序列已成功保存到文件: {output_path}")