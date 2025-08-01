import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_heap_sort_svl_v5(initial_array: list):
    """
    根据最终版 SVL 5.0 规范，为堆排序生成一个完整的可视化序列对象。
    此版本经过审查和重构，优化了 heapify 函数的可视化逻辑，并严格对齐所有规范。
    """
    
    # =================================================================
    # 1. 定义静态组件 (符合规范)
    # =================================================================
    
    algorithm_info = {
        "name": "Heap Sort",
        "family": "Sorting"
    }

    variables_schema = [
        {"name": "heap_size", "type": "value", "description": "当前堆的大小"},
        {"name": "i", "type": "pointer", "description": "当前操作的根节点索引"},
        {"name": "largest", "type": "pointer", "description": "父/子节点中最大值的索引"},
        {"name": "l", "type": "pointer", "description": "左子节点索引"},
        {"name": "r", "type": "pointer", "description": "右子节点索引"},
    ]
    
    pseudocode = [
        "function heapSort(array):",              # 1
        "  n = length(array)",                    # 2
        "  // Build max heap",                     # 3
        "  for i from n // 2 - 1 down to 0:",     # 4
        "    heapify(array, n, i)",               # 5
        "  // Heap extraction",                    # 6
        "  for i from n - 1 down to 1:",          # 7
        "    swap(array[0], array[i])",           # 8
        "    heapify(array, i, 0)",               # 9
        "",                                       # 10
        "function heapify(array, n, i):",        # 11
        "  largest = i",                          # 12
        "  l = 2*i + 1; r = 2*i + 2",             # 13
        "  if l < n and array[l] > array[largest]:", # 14
        "    largest = l",                        # 15
        "  if r < n and array[r] > array[largest]:", # 16
        "    largest = r",                        # 17
        "  if largest != i:",                     # 18
        "    swap(array[i], array[largest])",     # 19
        "    heapify(array, n, largest)",         # 20
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
        "data_state": {"type": "array", "data": initial_array_state},
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
    
    # --- Heapify 辅助函数 ---
    def heapify(heap_arr, heap_size, i, meta_base):
        largest = i
        l = 2 * i + 1
        r = 2 * i + 2
        
        meta = {**meta_base, "i": i, "largest": i, "l": "-", "r": "-"}
        deltas.append({"meta": meta, "code_highlight": 12, "operations": []})
        meta = {**meta, "l": l if l < heap_size else "-", "r": r if r < heap_size else "-"}
        deltas.append({"meta": meta, "code_highlight": 13, "operations": [
             {"op": "updateStyle", "params": {"indices": [i], "styleKey": "key_element"}}
        ]})

        # 比较父、左、右三个节点
        nodes_to_compare = [idx for idx in [i, l, r] if idx < heap_size]
        deltas.append({"meta": meta, "code_highlight": 14, "operations": [
            {"op": "updateStyle", "params": {"indices": nodes_to_compare, "styleKey": "compare"}}
        ]})

        if l < heap_size and heap_arr[l] > heap_arr[largest]:
            largest = l
        if r < heap_size and heap_arr[r] > heap_arr[largest]:
            largest = r
        
        meta["largest"] = largest
        deltas.append({"meta": meta, "code_highlight": 18, "operations": [
            {"op": "updateStyle", "params": {"indices": [largest], "styleKey": "key_element"}},
            {"op": "updateStyle", "params": {"indices": [idx for idx in nodes_to_compare if idx != largest], "styleKey": "heap_area"}}
        ]})

        if largest != i:
            deltas.append({"meta": meta, "code_highlight": 19, "operations": [
                {"op": "updateStyle", "params": {"indices": [i, largest], "styleKey": "swapping"}}
            ]})
            heap_arr[i], heap_arr[largest] = heap_arr[largest], heap_arr[i]
            deltas.append({"meta": meta, "code_highlight": 19, "operations": [
                {"op": "moveElements", "params": {"pairs": [{"fromIndex": i, "toIndex": largest}, {"fromIndex": largest, "toIndex": i}]}}
            ]})
            deltas.append({"meta": meta, "code_highlight": 19, "operations": [
                {"op": "updateStyle", "params": {"indices": [i, largest], "styleKey": "heap_area"}}
            ]})
            
            deltas.append({"meta": meta, "code_highlight": 20, "operations": []})
            heapify(heap_arr, heap_size, largest, meta_base)

    # --- 阶段一：建堆 (Build Max Heap) ---
    deltas.append({"meta": {}, "code_highlight": 3, "operations": [
        {"op": "updateStyle", "params": {"indices": list(range(n)), "styleKey": "heap_area"}}
    ]})
    deltas.append({"meta": {"heap_size": n}, "code_highlight": 4, "operations": [
        {"op": "updateBoundary", "params": {"type": "heap_boundary", "range": [0, n - 1], "label": "Build Max Heap", "styleKey": "heap_boundary"}}
    ]})
    for i in range(n // 2 - 1, -1, -1):
        deltas.append({"meta": {"heap_size": n, "i": i}, "code_highlight": 5, "operations": []})
        heapify(arr, n, i, meta_base={"heap_size": n})
        
    # --- 阶段二：堆排序 (Heap Extraction) ---
    deltas.append({"meta": {}, "code_highlight": 6, "operations": []})
    for i in range(n - 1, 0, -1):
        deltas.append({"meta": {"heap_size": i + 1, "i": i}, "code_highlight": 7, "operations": [
             {"op": "updateBoundary", "params": {"type": "heap_boundary", "range": [0, i], "styleKey": "heap_boundary"}}
        ]})
        
        deltas.append({"meta": {"heap_size": i + 1, "i": i}, "code_highlight": 8, "operations": [
            {"op": "updateStyle", "params": {"indices": [0, i], "styleKey": "swapping"}}
        ]})
        arr[0], arr[i] = arr[i], arr[0]
        deltas.append({"meta": {"heap_size": i + 1, "i": i}, "code_highlight": 8, "operations": [
            {"op": "moveElements", "params": {"pairs": [{"fromIndex": 0, "toIndex": i}, {"fromIndex": i, "toIndex": 0}]}}
        ]})
        
        deltas.append({"meta": {"heap_size": i, "i": i}, "code_highlight": 8, "operations": [
            {"op": "updateStyle", "params": {"indices": [i], "styleKey": "sorted"}},
            {"op": "updateStyle", "params": {"indices": [0], "styleKey": "heap_area"}},
            {"op": "updateBoundary", "params": {"type": "heap_boundary", "range": [0, i - 1], "styleKey": "heap_boundary"}}
        ]})
        
        deltas.append({"meta": {"heap_size": i, "i": 0}, "code_highlight": 9, "operations": []})
        heapify(arr, i, 0, meta_base={"heap_size": i})

    # --- 最终状态 ---
    deltas.append({"meta": {}, "code_highlight": 1, "operations": [
        {"op": "removeBoundary", "params": {"type": "heap_boundary"}},
        {"op": "updateStyle", "params": {"indices": list(range(n)), "styleKey": "sorted"}}
    ]})
    
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
    my_array = [4, 10, 3, 5, 1]
    output_filename = "heap_sort_svl_5.0.json"
    
    output_dir = Path("json_v5/sort")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"正在为数组 {my_array} 生成堆排序的SVL 5.0序列...")
    svl_output = generate_heap_sort_svl_v5(my_array)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (堆排序) 序列已成功保存到文件: {output_path}")