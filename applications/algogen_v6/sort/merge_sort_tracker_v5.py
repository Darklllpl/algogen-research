import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_merge_sort_svl_v5(initial_array: list):
    """
    根据最终版 SVL 5.0 规范，为归并排序生成一个完整的可视化序列对象。
    此版本使用 auxiliary_views 和动态视图操作来清晰地展示合并过程。
    """
    
    # =================================================================
    # 1. 定义静态组件 (符合规范)
    # =================================================================
    
    algorithm_info = {
        "name": "Merge Sort",
        "family": "Sorting"
    }

    variables_schema = [
        {"name": "l", "type": "pointer", "description": "左边界"},
        {"name": "m", "type": "pointer", "description": "中点"},
        {"name": "r", "type": "pointer", "description": "右边界"},
        {"name": "i", "type": "pointer", "description": "左子数组指针"},
        {"name": "j", "type": "pointer", "description": "右子数组指针"},
        {"name": "k", "type": "pointer", "description": "主数组合并指针"}
    ]
    
    pseudocode = [
        "function mergeSort(arr, l, r):",           # 1
        "  if l < r:",                                # 2
        "    m = l + (r - l) // 2",                 # 3
        "    mergeSort(arr, l, m)",                 # 4
        "    mergeSort(arr, m + 1, r)",             # 5
        "    merge(arr, l, m, r)",                  # 6
        "",                                         # 7
        "function merge(arr, l, m, r):",            # 8
        "  L = arr[l...m]; R = arr[m+1...r]",       # 9
        "  i=0, j=0, k=l",                          # 10
        "  while i < len(L) and j < len(R):",        # 11
        "    if L[i] <= R[j]: arr[k++] = L[i++]",  # 12
        "    else: arr[k++] = R[j++]",              # 13
        "  while i < len(L): arr[k++] = L[i++]",    # 14
        "  while j < len(R): arr[k++] = R[j++]",    # 15
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
        "auxiliary_views": [], # 初始时没有辅助视图
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
    
    # 使用一个显式的栈来模拟递归，(l, r, stage)，stage可以是 'split' 或 'merge'
    call_stack = [(0, n - 1, 'split')]
    
    def create_meta(l, m, r, i, j, k):
        vars = {"l": l, "m": m, "r": r, "i": i, "j": j, "k": k}
        return {key: val for key, val in vars.items() if val is not None and val != "-"}

    while call_stack:
        l, r, stage = call_stack.pop()
        
        if l >= r and stage == 'split':
            if 0 <= l < n:
                deltas.append({"meta": create_meta(l, None, r, None, None, None), "code_highlight": 2, "operations": [
                    {"op": "updateStyle", "params": {"indices": [l], "styleKey": "sorted"}}
                ]})
            continue

        m = l + (r - l) // 2

        if stage == 'split':
            deltas.append({"meta": create_meta(l, m, r, None, None, None), "code_highlight": 2, "operations": [
                {"op": "updateBoundary", "params": {"type": f"split_{l}_{r}", "range": [l, r], "label": f"Splitting", "styleKey": "partition_box"}}
            ]})
            
            # 推入栈的顺序是反向的：先推入merge, 再推右子树, 最后推左子树
            call_stack.append((l, r, 'merge'))
            deltas.append({"meta": create_meta(l, m, r, None, None, None), "code_highlight": 5, "operations": []})
            call_stack.append((m + 1, r, 'split'))
            deltas.append({"meta": create_meta(l, m, r, None, None, None), "code_highlight": 4, "operations": []})
            call_stack.append((l, m, 'split'))

        elif stage == 'merge':
            deltas.append({"meta": create_meta(l, m, r, None, None, None), "code_highlight": 6, "operations": [
                 {"op": "updateBoundary", "params": {"type": f"split_{l}_{r}", "range": [l, r], "label": f"Merging", "styleKey": "partition_box"}},
                 {"op": "updateStyle", "params": {"indices": list(range(l, r+1)), "styleKey": "sub_array_active"}}
            ]})
            
            L = arr[l : m + 1]
            R = arr[m + 1 : r + 1]
            view_id = f"merge_view_{l}_{r}"
            
            # >> Delta: 创建辅助视图来展示L和R <<
            aux_view = { "view_id": view_id, "type": "list", "title": f"Left & Right Sub-arrays", "data": {"L": L, "R": R} }
            deltas.append({"meta": create_meta(l, m, r, None, None, None), "code_highlight": 9, "operations": [
                {"op": "addAuxView", "params": {"view": aux_view}}
            ]})

            i, j, k = 0, 0, l
            deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 10, "operations": []})

            while i < len(L) and j < len(R):
                deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 11, "operations": []})
                
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 12, "operations": [
                        {"op": "updateValues", "params": {"updates": [{"index": k, "value": L[i]}]}},
                        {"op": "updateStyle", "params": {"indices": [k], "styleKey": "swapping"}}
                    ]})
                    i += 1
                else:
                    arr[k] = R[j]
                    deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 13, "operations": [
                        {"op": "updateValues", "params": {"updates": [{"index": k, "value": R[j]}]}},
                        {"op": "updateStyle", "params": {"indices": [k], "styleKey": "swapping"}}
                    ]})
                    j += 1
                k += 1
                deltas.append({"meta": create_meta(l,m,r,i,j,k-1), "code_highlight": 11, "operations": [
                    {"op": "updateStyle", "params": {"indices": [k-1], "styleKey": "sub_array_active"}}
                ]})


            while i < len(L):
                arr[k] = L[i]
                deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 14, "operations": [
                    {"op": "updateValues", "params": {"updates": [{"index": k, "value": L[i]}]}},
                    {"op": "updateStyle", "params": {"indices": [k], "styleKey": "swapping"}}
                ]})
                i += 1; k += 1
                deltas.append({"meta": create_meta(l,m,r,i,j,k-1), "code_highlight": 14, "operations": [
                    {"op": "updateStyle", "params": {"indices": [k-1], "styleKey": "sub_array_active"}}
                ]})
            
            while j < len(R):
                arr[k] = R[j]
                deltas.append({"meta": create_meta(l, m, r, i, j, k), "code_highlight": 15, "operations": [
                    {"op": "updateValues", "params": {"updates": [{"index": k, "value": R[j]}]}},
                    {"op": "updateStyle", "params": {"indices": [k], "styleKey": "swapping"}}
                ]})
                j += 1; k += 1
                deltas.append({"meta": create_meta(l,m,r,i,j,k-1), "code_highlight": 15, "operations": [
                    {"op": "updateStyle", "params": {"indices": [k-1], "styleKey": "sub_array_active"}}
                ]})
            
            # >> Delta: 合并完成, 移除辅助视图和边界, 标记为已排序 <<
            deltas.append({"meta": create_meta(l, None, r, None, None, None), "code_highlight": 6, "operations": [
                {"op": "removeAuxView", "params": {"view_id": view_id}},
                {"op": "removeBoundary", "params": {"type": f"split_{l}_{r}"}},
                {"op": "updateStyle", "params": {"indices": list(range(l, r + 1)), "styleKey": "sorted"}}
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
    my_array = [38, 27, 43, 3, 9, 82, 10]
    output_filename = "merge_sort_svl_5.0.json"
    
    output_dir = Path("json_v5/sort")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename
    
    print(f"正在为数组 {my_array} 生成归并排序的SVL 5.0序列...")
    svl_output = generate_merge_sort_svl_v5(my_array)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False)

    print(f"SVL 5.0 (归并排序) 序列已成功保存到文件: {output_path}")