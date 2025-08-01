import json
from pathlib import Path
import heapq
from collections import defaultdict
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_prim_svl_v5(graph_def: dict, start_node: str):
    """
    Generate complete visualization sequence for Prim algorithm according to SVL 5.0 specification.
    """
    
    # 1. Define static components
    
    algorithm_info = {
        "name": "Prim's Algorithm",
        "family": "Graph MST"
    }

    variables_schema = [
        {"name": "currentNode", "type": "value", "description": "Current node being added to MST"},
        {"name": "priorityQueue", "type": "list", "description": "Priority queue (key, node)"},
        {"name": "mst_cost", "type": "value", "description": "Current MST total weight"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "key", "type": "value", "description": "Minimum edge weight to connect to MST"},
            {"name": "parent", "type": "value", "description": "Parent node in MST"},
            {"name": "in_mst", "type": "flag", "description": "Whether already in MST"}
        ]
    }
    
    pseudocode = [
        "function Prim(graph, start):",
        "  key = init_keys_to_infinity()",
        "  parent = init_parents_to_null()",
        "  key[start] = 0",
        "  pq = new PriorityQueue(all_nodes)",
        "  while pq is not empty:",
        "    u = pq.extract_min()",
        "    // Add u to MST",
        "    for each neighbor v of u:",
        "      if v in pq and weight(u,v) < key[v]:",
        "        parent[v] = u",
        "        key[v] = weight(u,v)",
        "        pq.decrease_key(v, key[v])",
    ]

    # 2. Build initial frame
    
    all_nodes = list(graph_def.keys())
    nodes = [{"id": n, "styleKey": "idle_node", "properties": {"key": "∞", "parent": "-", "in_mst": False}} for n in all_nodes]
    
    edges, edge_set = [], set()
    for u, neighbors in graph_def.items():
        for v, weight in neighbors:
            if tuple(sorted((u, v))) not in edge_set:
                edges.append({"from": u, "to": v, "directed": False, "label": str(weight), "styleKey": "normal_edge", "properties": {"weight": weight}})
                edge_set.add(tuple(sorted((u, v))))
    
    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "graph", "structure": {"nodes": nodes, "edges": edges}},
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    
    deltas = []
    # Initialize algorithm data structures
    key = {node_id: float('inf') for node_id in all_nodes}
    parent = {node_id: None for node_id in all_nodes}
    in_mst = {node_id: False for node_id in all_nodes}
    pq = [] # (key, node_id)
    
    key[start_node] = 0
    # In practice, usually only put start node in pq, or build heap for all nodes. Here we simulate the former.
    heapq.heappush(pq, (0, start_node))
    
    # Maintain node properties copy in memory
    node_props = {n["id"]: n["properties"] for n in initial_frame["data_state"]["structure"]["nodes"]}
    node_props[start_node]["key"] = 0
    
    deltas.append({"meta": {}, "code_highlight": 2, "operations": []})
    deltas.append({"meta": {"priorityQueue": sorted(pq)}, "code_highlight": 4, "operations": [
        {"op": "updateNodeProperties", "params": {"updates": [{"id": start_node, "properties": {"key": 0}}]}}
    ]})
    
    mst_cost = 0
    
    while pq:
        pq_display = sorted(pq) 
        deltas.append({"meta": {"priorityQueue": pq_display, "mst_cost": mst_cost}, "code_highlight": 6, "operations": []})
        
        weight_u, u = heapq.heappop(pq)
        
        if in_mst[u]:
            continue
            
        in_mst[u] = True
        node_props[u]["in_mst"] = True
        mst_cost += weight_u

        # >> Delta: 节点u出队，正式加入MST <<
        u_parent = parent[u]
        ops = [
            {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "in_path_node"}},
            {"op": "updateNodeProperties", "params": {"updates": [{"id": u, "properties": {"in_mst": True}}]}}
        ]
        # 如果不是起点，高亮连接它的边
        if u_parent is not None:
            ops.append({"op": "updateEdgeStyle", "params": {"edges": [{"from": u_parent, "to": u}], "styleKey": "in_path_edge"}})
            
        deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 8, "operations": ops})
        
        deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 9, "operations": []})

        for v, weight_uv in graph_def.get(u, []):
            if not in_mst[v]:
                # >> Delta: 高亮正在考察的邻居和边 <<
                deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 10, "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "compare"}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "traversed_edge"}}
                ]})

                if weight_uv < key[v]:
                    # >> Delta: 更新邻居的key和parent <<
                    key[v] = weight_uv
                    parent[v] = u
                    node_props[v]["key"] = weight_uv
                    node_props[v]["parent"] = u
                    heapq.heappush(pq, (weight_uv, v))
                    
                    deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 11, "operations": []})
                    deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 12, "operations": [
                         {"op": "updateNodeProperties", "params": {"updates": [{"id": v, "properties": {"key": weight_uv, "parent": u}}]}}
                    ]})

                # >> Delta: 恢复考察过的邻居和边的样式 <<
                deltas.append({"meta": {"priorityQueue": sorted(pq), "currentNode": u, "mst_cost": mst_cost}, "code_highlight": 9, "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "idle_node"}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
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
    # Example weighted undirected graph adjacency list
    graph = {
        'A': [('B', 2), ('D', 5)],
        'B': [('A', 2), ('C', 4), ('D', 1), ('E', 3)],
        'C': [('B', 4), ('E', 6)],
        'D': [('A', 5), ('B', 1), ('E', 5)],
        'E': [('B', 3), ('C', 6), ('D', 5)]
    }
    start_node = 'A'
    
    output_filename = "prim_svl_5.0.json"
    output_dir = Path("json_v5/graph")
    output_dir.mkdir(exist_ok=True) 
    output_path = output_dir / output_filename

    print(f"Generating Prim algorithm SVL 5.0 sequence for graph (start: {start_node})...")
    svl_output = generate_prim_svl_v5(graph, start_node)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (Prim) sequence saved to: {output_path}")