import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_quicksort_svl_v5(initial_array: list):
    """
    根据最终版 SVL 5.0 规范，为快速排序 (Lomuto 分区方案) 生成一个完整的可视化序列。
    此版本严格对齐所有最终规范和标准样式。
    """
    
    # =================================================================
    # 1. 定义静态组件 (符合规范)
    # =================================================================
    
    algorithm_info = {
        "name": "Quick Sort (Lomuto Partition)",
        "family": "Sorting"
    }

    variables_schema = [
        {"name": "low", "type": "pointer", "description": "当前处理分区的起始索引"},
        {"name": "high", "type": "pointer", "description": "当前处理分区的结束索引"},
        {"name": "pivot", "type": "value", "description": "当前分区的主元值"},
        {"name": "i", "type": "pointer", "description": "小于主元区域的边界"},
        {"name": "j", "type": "pointer", "description": "当前扫描的元素索引"}
    ]
    
    pseudocode = [
        "function quickSort(array, low, high):",       # 1
        "  if low < high:",                            # 2
        "    pi = partition(array, low, high)",        # 3
        "    quickSort(array, low, pi - 1)",           # 4
        "    quickSort(array, pi + 1, high)",          # 5
        "",                                            # 6
        "function partition(array, low, high):",       # 7
        "  pivot = array[high]",                       # 8
        "  i = low - 1",                               # 9
        "  for j from low to high - 1:",               # 10
        "    if array[j] < pivot:",                    # 11
        "      i = i + 1",                             # 12
        "      swap(array[i], array[j])",              # 13
        "  swap(array[i+1], array[high])",             # 14
        "  return i + 1"                               # 15
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
        "data_state": { "type": "array", "data": initial_array_state },
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
    
    # 使用一个显式的栈来模拟递归，便于追踪
    call_stack = [(0, n - 1)]
    
    def create_meta(low, high, pivot, i, j):
        return {"low": low, "high": high, "pivot": pivot, "i": i, "j": j}

    while call_stack:
        low, high = call_stack.pop(0)
        
        deltas.append({"meta": create_meta(low, high, "-", "-", "-"), "code_highlight": 2, "operations": []})
        if low >= high:
            if 0 <= low < n:
                 # 单个元素的子数组直接标记为已排序
                 deltas.append({"meta": {}, "code_highlight": 2, "operations": [
                     {"op": "updateStyle", "params": {"indices": [low], "styleKey": "sorted"}}
                 ]})
            continue

        # --- 分区开始 ---
        deltas.append({
            "meta": create_meta(low, high, "-", "-", "-"), "code_highlight": 3,
            "operations": [[
                {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                {"op": "updateStyle", "params": {"indices": list(range(low, high + 1)), "styleKey": "sub_array_active"}}
            ]]
        })
        
        # --- partition 函数逻辑 ---
        pivot = arr[high]
        i = low - 1
        
        deltas.append({"meta": create_meta(low, high, pivot, i, "-"), "code_highlight": 8, "operations": [
            {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
            {"op": "updateStyle", "params": {"indices": [high], "styleKey": "pivot"}}
        ]})
        deltas.append({"meta": create_meta(low, high, pivot, i, "-"), "code_highlight": 9, "operations": [
            {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}}
        ]})

        for j in range(low, high):
            deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 10, "operations": [
                {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                {"op": "updateStyle", "params": {"indices": [j], "styleKey": "compare"}}
            ]})
            deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 11, "operations": [
                {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}}
            ]})
            
            if arr[j] < pivot:
                i += 1
                deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 12, "operations": [
                    {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                    {"op": "updateStyle", "params": {"indices": list(range(low, i + 1)), "styleKey": "partition_area"}}
                ]})
                
                if i != j:
                    deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 13, "operations": [
                        {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                        {"op": "updateStyle", "params": {"indices": [i, j], "styleKey": "swapping"}}
                    ]})
                    
                    arr[i], arr[j] = arr[j], arr[i]
                    deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 13, "operations": [
                        {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                        {"op": "moveElements", "params": {"pairs": [{"fromIndex": i, "toIndex": j}, {"fromIndex": j, "toIndex": i}]}}
                    ]})
                
                # 恢复样式
                deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 13, "operations": [
                    {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                    {"op": "updateStyle", "params": {"indices": [i], "styleKey": "partition_area"}},
                    {"op": "updateStyle", "params": {"indices": [j], "styleKey": "sub_array_active"}}
                ]})
            else:
                 deltas.append({"meta": create_meta(low, high, pivot, i, j), "code_highlight": 11, "operations": [
                    {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
                    {"op": "updateStyle", "params": {"indices": [j], "styleKey": "sub_array_active"}}
                ]})

        # --- 将 pivot 换到正确位置 ---
        pi = i + 1
        deltas.append({"meta": create_meta(low, high, pivot, i, "-"), "code_highlight": 14, "operations": [
            {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
            {"op": "updateStyle", "params": {"indices": [pi, high], "styleKey": "swapping"}}
        ]})

        arr[pi], arr[high] = arr[high], arr[pi]
        deltas.append({"meta": create_meta(low, high, pivot, i, "-"), "code_highlight": 14, "operations": [
            {"op": "updateBoundary", "params": {"type": "partition_box", "range": [low, high], "label": f"Partition({low}, {high})", "styleKey": "partition_box"}},
            {"op": "moveElements", "params": {"pairs": [{"fromIndex": pi, "toIndex": high}, {"fromIndex": high, "toIndex": pi}]}}
        ]})

        # --- 分区结束 ---
        deltas.append({"meta": create_meta(low, high, "-", "-", "-"), "code_highlight": 15, "operations": [
            {"op": "removeBoundary", "params": {"type": "partition_box"}},
            {"op": "updateStyle", "params": {"indices": list(range(low, high + 1)), "styleKey": "idle"}},
            {"op": "updateStyle", "params": {"indices": [pi], "styleKey": "sorted"}},
        ]})

        # 将新的子任务推入栈中，LIFO顺序确保先处理小的子问题
        call_stack.insert(0, (pi + 1, high))
        deltas.append({"meta": {}, "code_highlight": 5, "operations": []})

        call_stack.insert(0, (low, pi - 1))
        deltas.append({"meta": {}, "code_highlight": 4, "operations": []})

    deltas.append({"meta": {}, "code_highlight": 1, "operations": []})
    
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
    my_array = [8, 2, 9, 7]
    output_filename = "quicksort_svl_5.0.json"
    
    output_dir = Path("json_v5/sort")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    
    print(f"正在为数组 {my_array} 生成快速排序的SVL 5.0序列...")
    svl_output = generate_quicksort_svl_v5(my_array)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False)

    print(f"SVL 5.0 (快速排序) 序列已成功保存到文件: {output_path}")