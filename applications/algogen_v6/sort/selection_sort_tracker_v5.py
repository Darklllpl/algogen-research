import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_selection_sort_svl_v5(initial_array: list):
    """
    根据最终版 SVL 5.0 规范，为选择排序生成一个完整的可视化序列对象。
    此版本使用标准化的样式（如 key_element, swapping）以增强可视化清晰度。
    """
    
    # =================================================================
    # 1. 定义静态组件 (符合规范)
    # =================================================================
    
    algorithm_info = {
        "name": "Selection Sort",
        "family": "Sorting"
    }

    variables_schema = [
        {"name": "i", "type": "pointer", "description": "主循环指针, 标记已排序区域的边界"},
        {"name": "j", "type": "pointer", "description": "内层循环指针, 用于寻找最小元素"},
        {"name": "min_idx", "type": "pointer", "description": "当前找到的最小元素的索引"}
    ]
    
    pseudocode = [
        "function SelectionSort(array):",        # 1
        "  n = length(array)",                  # 2
        "  for i from 0 to n-1:",               # 3
        "    min_idx = i",                      # 4
        "    for j from i+1 to n:",             # 5
        "      if array[j] < array[min_idx]:",   # 6
        "        min_idx = j",                  # 7
        "    swap(array[i], array[min_idx])",     # 8
        "  return array"                        # 9
    ]

    # =================================================================
    # 2. 构建初始帧 (initial_frame)
    # =================================================================
    
    initial_array_state = [
        {"index": i, "value": val, "state": "idle"}
        for i, val in enumerate(initial_array)
    ]
    
    initial_frame = {
        "data_schema": {},
        "data_state": {
            "type": "array",
            "data": initial_array_state
        },
        "variables_schema": variables_schema,
        "pseudocode": pseudocode,
        "code_highlight": 1,
        "styles": DEFAULT_STYLES
    }

    # =================================================================
    # 3. 追踪算法执行，生成Deltas
    # =================================================================
    
    deltas = []
    arr = list(initial_array)
    n = len(arr)
    
    deltas.append({"meta": {}, "code_highlight": 2, "operations": []})
    deltas.append({"meta": {}, "code_highlight": 3, "operations": []})
    for i in range(n):
        min_idx = i
        
        # >> Delta: 更新外层循环变量i, 并标记当前最小元素 <<
        deltas.append({
            "meta": {"i": i, "min_idx": min_idx, "j": "-"}, 
            "code_highlight": 4, 
            "operations": [
                {"op": "updateStyle", "params": {"indices": [min_idx], "styleKey": "key_element"}}
            ]
        })
        
        deltas.append({"meta": {"i": i, "min_idx": min_idx, "j": i+1}, "code_highlight": 5, "operations": []})
        for j in range(i + 1, n):
            current_meta = {"i": i, "j": j, "min_idx": min_idx}
            
            # >> Delta: 高亮比较 j 和 min_idx <<
            # min_idx 保持 key_element 样式, j 使用 compare 样式
            deltas.append({
                "meta": current_meta,
                "code_highlight": 6,
                "operations": [
                    {"op": "updateStyle", "params": {"indices": [j], "styleKey": "compare"}}
                ]
            })

            if arr[j] < arr[min_idx]:
                # >> Delta: 发现新的最小元素，更新 min_idx <<
                old_min_idx = min_idx
                min_idx = j
                current_meta["min_idx"] = min_idx # 更新 meta
                
                deltas.append({
                    "meta": current_meta,
                    "code_highlight": 7,
                    "operations": [
                        [
                            # 旧的 min_idx 恢复 idle 状态
                            {"op": "updateStyle", "params": {"indices": [old_min_idx], "styleKey": "idle"}},
                            # 新的 min_idx (即j) 更新为 key_element 样式
                            {"op": "updateStyle", "params": {"indices": [min_idx], "styleKey": "key_element"}}
                        ]
                    ]
                })
            else:
                 # >> Delta: 未发现新最小元素, 仅恢复 j 的样式 <<
                 deltas.append({
                    "meta": current_meta,
                    "code_highlight": 6, # 仍在判断行，表示判断为false
                    "operations": [
                        {"op": "updateStyle", "params": {"indices": [j], "styleKey": "idle"}}
                    ]
                })

        # 内层循环结束
        # >> Delta: 准备交换 i 和最终找到的 min_idx <<
        if i != min_idx:
            deltas.append({
                "meta": {"i": i, "j": "-", "min_idx": min_idx},
                "code_highlight": 8,
                "operations": [
                    # 使用 'swapping' 样式高亮即将交换的两个元素
                    {"op": "updateStyle", "params": {"indices": [i, min_idx], "styleKey": "swapping"}}
                ]
            })
            
            # >> Delta: 执行交换 <<
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
            deltas.append({
                "meta": {"i": i, "j": "-", "min_idx": min_idx},
                "code_highlight": 8,
                "operations": [
                    {"op": "moveElements", "params": {
                        "pairs": [{"fromIndex": i, "toIndex": min_idx}, {"fromIndex": min_idx, "toIndex": i}]
                    }}
                ]
            })
        
        # >> Delta: 将i位置标记为已排序，并恢复另一个元素的样式 <<
        ops = [{"op": "updateStyle", "params": {"indices": [i], "styleKey": "sorted"}}]
        if i != min_idx:
            # 如果交换过，恢复被交换元素为 idle
            ops.append({"op": "updateStyle", "params": {"indices": [min_idx], "styleKey": "idle"}})
        deltas.append({
            "meta": {"i": i, "j": "-", "min_idx": "-"},
            "code_highlight": 3,  # 回到下一次 i 循环
            "operations": [ops]
        })
            
    # >> Delta: 排序结束 <<
    deltas.append({"meta": {}, "code_highlight": 9, "operations": []})

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
    my_array = [64, 25, 12, 22, 11]
    output_filename = "selection_sort_svl_5.0.json"
    
    output_dir = Path("json_v5/sort")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"正在为数组 {my_array} 生成选择排序的SVL 5.0序列...")
    svl_output = generate_selection_sort_svl_v5(my_array)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (选择排序) 序列已成功保存到文件: {output_path}")