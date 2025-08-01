import json
from pathlib import Path
import heapq
from collections import defaultdict
from copy import deepcopy
import sys
import os

def build_initial_aux_table(node_props):
    """
    Build initial auxiliary table for frame.
    """
    nodes = sorted(node_props.keys())
    dist_row = []
    pred_row = []
    for n in nodes:
        val = node_props[n]['dist']
        # Handle infinity symbol display
        if isinstance(val, float) and val == float('inf'):
            dist_row.append(r"$\infty$")
        elif val == "∞" or val == r"$\infty$":
            dist_row.append(r"$\infty$")
        else:
            dist_row.append(str(val))
        
        pred_val = node_props[n]['pred']
        pred_row.append(str(pred_val) if pred_val not in (None, "-", "") else "-")
    
    return {
        "view_id": "dijkstra_table",
        "type": "table",
        "title": "Distance / Predecessor Table",
        "data": [
            ["Node"] + nodes,
            ["dist"] + dist_row,
            ["pred"] + pred_row
        ]
    }

def make_frame(meta, code_highlight, operations):
    """
    Build standard FrameDelta object without auxiliary_views.
    """
    return {
        "meta": deepcopy(meta),
        "code_highlight": code_highlight,
        "operations": deepcopy(operations),
    }

def generate_dijkstra_svl_v5(graph_def: dict, start_node: str, styles: dict):
    algorithm_info = {
        "name": "Dijkstra's Algorithm",
        "family": "Graph Shortest Path"
    }

    variables_schema = [
        {"name": "currentNode", "type": "value", "description": "Node being popped from PQ"},
        {"name": "priorityQueue", "type": "list", "description": "Priority queue (dist, node)"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "dist", "type": "value", "description": "Distance from start node"},
            {"name": "pred", "type": "value", "description": "Predecessor node"}
        ]
    }

    pseudocode = [
        "function Dijkstra(graph, start):",                            # 1
        "  dist = init_distances_to_infinity()",                      # 2
        "  dist[start] = 0",                                          # 3
        "  pq = new PriorityQueue()",                                 # 4
        "  pq.add((0, start))",                                       # 5
        "  while pq is not empty:",                                   # 6
        "    dist_u, u = pq.pop()",                                   # 7
        "    if dist_u > dist[u]: continue",                           # 8
        "    for each neighbor v of u:",                              # 9
        "      weight_uv = weight(u, v)",                             # 10
        "      if dist[u] + weight_uv < dist[v]:",                     # 11
        "        dist[v] = dist[u] + weight_uv",                      # 12
        "        predecessor[v] = u",                                 # 13
        "        pq.add((dist[v], v))",                               # 14
    ]

    # Initial state
    nodes = [{"id": n, "styleKey": "idle_node", "properties": {"dist": "∞", "pred": "-"}} for n in graph_def.keys()]
    edges, edge_set = [], set()
    for u, neighbors in graph_def.items():
        for v, weight in neighbors:
            key = (min(u,v), max(u,v))
            if key not in edge_set:
                edges.append({"from": u, "to": v, "directed": False, "label": str(weight), "styleKey": "normal_edge", "properties": {"weight": weight}})
                edge_set.add(key)
    
    node_props = {n["id"]: deepcopy(n["properties"]) for n in nodes}
    
    sorted_node_ids = sorted(graph_def.keys())
    node_to_col_idx = {node_id: i + 1 for i, node_id in enumerate(sorted_node_ids)}

    initial_aux_table = build_initial_aux_table(node_props)
    
    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "graph", "structure": {"nodes": deepcopy(nodes), "edges": edges}},
        "auxiliary_views": [deepcopy(initial_aux_table)],
        "variables_schema": variables_schema,
        "pseudocode": pseudocode, 
        "code_highlight": 1,
        "styles": styles
    }

    # ========== 追踪算法 ==========
    deltas = []
    distances = {node['id']: float('inf') for node in nodes}
    predecessors = {node['id']: None for node in nodes}
    pq = []
    
    deltas.append(make_frame({}, 2, []))

    distances[start_node] = 0
    node_props[start_node]["dist"] = 0
    
    start_node_col = node_to_col_idx[start_node]
    update_dist_op = { "op": "updateTableCell", "params": {"view_id": "dijkstra_table", "updates": [{"row": 1, "col": start_node_col, "value": 0}]} }
    update_node_prop_op = {"op": "updateNodeProperties", "params": {"updates": [{"id": start_node, "properties": {"dist": 0}}]}}
    deltas.append(make_frame({}, 3, [update_node_prop_op, update_dist_op]))

    deltas.append(make_frame({}, 4, []))
    heapq.heappush(pq, (0, start_node))
    deltas.append(make_frame({"priorityQueue": sorted(pq)}, 5, []))

    visited_nodes = set()

    while pq:
        pq_display_before_pop = sorted(pq)
        deltas.append(make_frame({"priorityQueue": pq_display_before_pop}, 6, []))
        
        dist_u, u = heapq.heappop(pq)
        pq_display_after_pop = sorted(pq)

        path_nodes, path_edges = [], []
        curr = u
        while curr is not None and predecessors[curr] is not None:
            pred = predecessors[curr]
            path_nodes.append(curr)
            path_edges.append({"from": pred, "to": curr})
            curr = pred
        if curr is not None: path_nodes.append(curr)

        deltas.append(make_frame(
            {"priorityQueue": pq_display_after_pop, "currentNode": u}, 7, [
                {"op": "updateNodeStyle", "params": {"ids": list(visited_nodes), "styleKey": "visited_node"}},
                {"op": "updateNodeStyle", "params": {"ids": [n for n in path_nodes if n not in visited_nodes], "styleKey": "in_path_node"}},
                {"op": "updateEdgeStyle", "params": {"edges": path_edges, "styleKey": "in_path_edge"}},
                {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "current_node"}},
            ]
        ))

        if dist_u > distances[u]:
            deltas.append(make_frame({"priorityQueue": pq_display_after_pop, "currentNode": u}, 8, []))
            continue
        
        visited_nodes.add(u)
        
        # 【修正】: 移除此处的 deltas.append 调用。
        # 不再在开始遍历邻居前就将 currentNode 变灰，让它保持黄色高亮。
        # deltas.append(make_frame(
        #     {"priorityQueue": pq_display_after_pop, "currentNode": u}, 9, [
        #         {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "visited_node"}}
        #     ]
        # ))
        
        # 【新增】: 添加一个标志，判断是否已经为开始遍历邻居创建了初始帧
        added_neighbor_loop_frame = False

        for v, weight in graph_def.get(u, []):
            if v in visited_nodes:
                continue

            # 【修正】: 确保 "for each neighbor" (第9行) 的帧只在第一次循环时添加一次
            if not added_neighbor_loop_frame:
                deltas.append(make_frame({"priorityQueue": pq_display_after_pop, "currentNode": u}, 9, []))
                added_neighbor_loop_frame = True
            
            deltas.append(make_frame({"priorityQueue": pq_display_after_pop, "currentNode": u}, 10, []))
            
            deltas.append(make_frame(
                {"priorityQueue": pq_display_after_pop, "currentNode": u}, 11, [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "compare"}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "traversed_edge"}}
                ]
            ))

            if distances[u] + weight < distances[v]:
                new_dist_v = distances[u] + weight
                
                distances[v] = new_dist_v
                predecessors[v] = u
                node_props[v]["dist"] = new_dist_v
                node_props[v]["pred"] = u
                
                v_col = node_to_col_idx[v]
                update_table_ops = {
                    "op": "updateTableCell",
                    "params": {
                        "view_id": "dijkstra_table",
                        "updates": [ {"row": 1, "col": v_col, "value": round(new_dist_v, 2)}, {"row": 2, "col": v_col, "value": u} ]
                    }
                }
                update_node_prop_op_v = {"op": "updateNodeProperties", "params": {"updates": [{"id": v, "properties": {"dist": new_dist_v, "pred": u}}]}}
                
                deltas.append(make_frame({"priorityQueue": pq_display_after_pop, "currentNode": u}, 12, []))
                deltas.append(make_frame({"priorityQueue": pq_display_after_pop, "currentNode": u}, 13, [update_node_prop_op_v, update_table_ops]))

                heapq.heappush(pq, (new_dist_v, v))
                pq_display_after_push = sorted(pq)
                deltas.append(make_frame({"priorityQueue": pq_display_after_push, "currentNode": u}, 14, []))

            final_style = "visited_node" if v in visited_nodes else "idle_node"
            deltas.append(make_frame(
                {"priorityQueue": sorted(pq), "currentNode": u}, 9, [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": final_style}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
                ]
            ))

        # 【新增】: 在 currentNode (u) 的所有邻居都处理完毕后, 才将其样式从 "current_node" 更新为 "visited_node"
        deltas.append(make_frame(
            {"priorityQueue": sorted(pq), "currentNode": u}, 6, [
                {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "visited_node"}}
            ]
        ))
            
    final_path_nodes, final_path_edges = [], []
    for node_id in sorted_node_ids:
        curr = node_id
        while curr is not None and predecessors[curr] is not None:
            pred = predecessors[curr]
            final_path_nodes.append(curr)
            final_path_edges.append({"from": pred, "to": curr})
            curr = pred
    final_ops = [
        {"op": "updateNodeStyle", "params": {"ids": list(visited_nodes), "styleKey": "visited_node"}},
        {"op": "updateEdgeStyle", "params": {"edges": final_path_edges, "styleKey": "in_path_edge"}}
    ]
    deltas.append(make_frame({}, -1, final_ops))

    svl_obj = { "svl_version": "5.0", "algorithm": algorithm_info, "initial_frame": initial_frame, "deltas": deltas }
    return svl_obj

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_script_dir)
    sys.path.insert(0, parent_dir)
    
    try: from default_styles import DEFAULT_STYLES
    except ImportError:
        DEFAULT_STYLES = {
            "elementStyles": {
                "idle_node": {"fill": "#FFFFFF", "stroke": "#424242", "labelColor": "#000000"}, "visited_node": {"fill": "#E0E0E0", "stroke": "#616161"},
                "current_node": {"fill": "#FFC107", "stroke": "#FF8F00"}, "compare": {"fill": "#29B6F6", "stroke": "#0288D1"},
                "in_path_node": {"fill": "#D1C4E9", "stroke": "#512DA8"}, "normal_edge": {"stroke": "#9E9E9E", "strokeWidth": 2},
                "traversed_edge": {"stroke": "#03A9F4", "strokeWidth": 4}, "in_path_edge": {"stroke": "#7E57C2", "strokeWidth": 4}
            }
        }

    graph = {
        'A': [('B', 4), ('C', 3), ('D', 10)], 'B': [('A', 4), ('C', 2), ('E', 5)],
        'C': [('A', 3), ('B', 2), ('D', 6), ('F', 7)], 'D': [('A', 10), ('C', 6), ('G', 3)],
        'E': [('B', 5), ('F', 1)], 'F': [('C', 7), ('E', 1), ('G', 8)], 'G': [('D', 3), ('F', 8)]
    }
    start_node = 'A'
    
    output_dir = Path("json_v5/graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dijkstra_svl_5.0.json"
    
    svl = generate_dijkstra_svl_v5(graph, start_node, styles=DEFAULT_STYLES)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(svl, f, indent=2, ensure_ascii=False)
        
    print(f"SVL 5.0 with corrected visualization logic saved to: {output_path}")