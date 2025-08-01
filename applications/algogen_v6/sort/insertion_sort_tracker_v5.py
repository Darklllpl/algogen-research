import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_insertion_sort_svl_v5(initial_array: list):
    """
    根据最终版 SVL 5.0 规范，为插入排序生成一个完整的可视化序列对象。
    此版本保留了“提起-平移-放回”的可视化逻辑，并严格对齐所有规范。
    """
    
    # =================================================================
    # 1. 定义静态组件 (符合规范)
    # =================================================================
    
    algorithm_info = {
        "name": "Insertion Sort",
        "family": "Sorting"
    } #

    variables_schema = [
        {"name": "i", "type": "pointer", "description": "主循环指针, 标记未排序部分的开始"},
        {"name": "j", "type": "pointer", "description": "内层循环指针, 在已排序部分中扫描"},
        {"name": "key", "type": "value", "description": "当前待插入的元素值"}
    ] #
    
    pseudocode = [
        "function InsertionSort(array):",              # 1
        "  for i from 1 to n-1:",                     # 2
        "    key = array[i]",                         # 3
        "    j = i - 1",                              # 4
        "    while j >= 0 and key < array[j]:",        # 5
        "      array[j+1] = array[j]",                # 6
        "      j = j - 1",                            # 7
        "    array[j+1] = key",                       # 8
        "  return array"                              # 9
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
        }, #
        "variables_schema": variables_schema,
        "pseudocode": pseudocode,
        "code_highlight": 1,
        "styles": DEFAULT_STYLES #
    } #

    # =================================================================
    # 3. 追踪算法执行，生成Deltas
    # =================================================================
    
    deltas = []
    arr = list(initial_array)
    n = len(arr)
    
    # 第一个元素默认已排序
    deltas.append({"meta": {}, "code_highlight": 2, "operations": [
        {"op": "updateStyle", "params": {"indices": [0], "styleKey": "sorted"}}
    ]})

    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        # >> Delta: “提起” Key 元素 <<
        deltas.append({
            "meta": {"i": i, "key": key, "j": "-"},
            "code_highlight": 3,
            "operations": [
                [
                    {"op": "drawTemp", "params": {"type": "key_holder", "value": key, "styleKey": "key_holder_box"}},
                    {"op": "updateStyle", "params": {"indices": [i], "styleKey": "placeholder"}}
                ]
            ]
        }) #
        
        deltas.append({"meta": {"i": i, "key": key, "j": j}, "code_highlight": 4, "operations": []}) 
        
        while j >= 0 and key < arr[j]:
            current_meta = {"i": i, "key": key, "j": j}
            
            # >> Delta: 比较 Key 和 j 位置的元素 <<
            deltas.append({"meta": current_meta, "code_highlight": 5, "operations": [
                {"op": "updateStyle", "params": {"indices": [j], "styleKey": "compare"}}
            ]})

            # >> Delta: “平移” j 位置的元素到 j+1 <<
            arr[j + 1] = arr[j] # 实际更新数组
            deltas.append({
                "meta": current_meta, "code_highlight": 6,
                "operations": [
                    [
                        {"op": "shiftElements", "params": {"shifts": [{"fromIndex": j, "toIndex": j + 1, "value": arr[j]}]}},
                        {"op": "updateStyle", "params": {"indices": [j], "styleKey": "placeholder"}}, # 原位置变坑位
                        {"op": "updateStyle", "params": {"indices": [j+1], "styleKey": "shifting"}}
                    ]
                ]
            })

            # >> Delta: 恢复平移后元素的样式 <<
            deltas.append({"meta": current_meta, "code_highlight": 6, "operations": [
                 {"op": "updateStyle", "params": {"indices": [j+1], "styleKey": "sorted"}}
            ]})

            j -= 1
            
            deltas.append({"meta": {"i": i, "key": key, "j": j}, "code_highlight": 7, "operations": []})
            
        # >> Delta: while循环结束的最终条件检查 <<
        deltas.append({"meta": {"i": i, "key": key, "j": j}, "code_highlight": 5, "operations": []})
            
        # >> Delta: “放回” Key 元素 <<
        arr[j + 1] = key # 实际更新数组
        deltas.append({
            "meta": {"i": i, "key": key, "j": j}, "code_highlight": 8,
            "operations": [
                [
                    {"op": "removeTemp", "params": {"type": "key_holder"}},
                    {"op": "updateValues", "params": {"updates": [{"index": j + 1, "value": key}]}},
                    # 放回后，整个已排序区域都应是sorted状态
                    {"op": "updateStyle", "params": {"indices": list(range(i + 1)), "styleKey": "sorted"}}
                ]
            ]
        })
        
        # >> Delta: 准备下一次 i 循环 <<
        deltas.append({"meta": {"i": i, "key": "-", "j": "-"}, "code_highlight": 2, "operations": []})
        
    # >> Delta: 最终状态 <<
    deltas.append({"meta": {}, "code_highlight": 9, "operations": [
        {"op": "updateStyle", "params": {"indices": list(range(n)), "styleKey": "sorted"}}
    ]})

    # =================================================================
    # 4. 组装并返回最终的SVL对象
    # =================================================================
    final_svl_object = {
        "svl_version": "5.0", #
        "algorithm": algorithm_info,
        "initial_frame": initial_frame,
        "deltas": deltas
    }
    return final_svl_object

if __name__ == '__main__':
    my_array = [5, 2, 4, 6, 1, 3]
    output_filename = "insertion_sort_svl_5.0.json"
    
    output_dir = Path("json_v5/sort") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"正在为数组 {my_array} 生成插入排序的SVL 5.0序列...")
    svl_output = generate_insertion_sort_svl_v5(my_array)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (插入排序) 序列已成功保存到文件: {output_path}")