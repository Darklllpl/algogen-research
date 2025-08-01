import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_bellman_ford_svl_v5(graph_def: dict, start_node: str):
    """
    Generate complete visualization sequence for Bellman-Ford algorithm according to SVL 5.0 specification.
    This version clearly shows V-1 relaxation rounds and final negative cycle detection.
    """
    
    # 1. Define static components
    
    algorithm_info = {
        "name": "Bellman-Ford Algorithm",
        "family": "Graph Shortest Path"
    }

    variables_schema = [
        {"name": "iteration", "type": "value", "description": "Relaxation pass"},
        {"name": "edge", "type": "value", "description": "Current edge being relaxed"},
        {"name": "status", "type": "value", "description": "Algorithm status"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "dist", "type": "value", "description": "Distance from start"},
            {"name": "pred", "type": "value", "description": "Predecessor node"}
        ]
    }
    
    pseudocode = [
        "function BellmanFord(graph, start):",
        "  dist = init_distances_to_infinity()",
        "  dist[start] = 0",
        "  // V-1 Relaxation Passes",
        "  for i from 1 to V-1:",
        "    for each edge (u, v, w) in graph:",
        "      if dist[u] + w < dist[v]:",
        "        dist[v] = dist[u] + w",
        "        predecessor[v] = u",
        "  // Check for negative-weight cycles",
        "  for each edge (u, v, w) in graph:",
        "    if dist[u] + w < dist[v]:",
        "      return 'Negative Cycle Detected'",
        "  return dist, predecessor",
    ]

    # 2. Build initial frame
    
    all_nodes_ids = list(graph_def.keys())
    nodes = [{"id": n, "styleKey": "idle_node", "properties": {"dist": "∞", "pred": "-"}} for n in all_nodes_ids]
    
    all_edges = []
    for u, neighbors in graph_def.items():
        for v, weight in neighbors:
            # Bellman-Ford handles directed edges
            all_edges.append({"from": u, "to": v, "directed": True, "label": str(weight), "styleKey": "normal_edge", "properties": {"weight": weight}})

    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "graph", "structure": {"nodes": nodes, "edges": all_edges}},
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    
    deltas = []
    num_nodes = len(all_nodes_ids)
    distances = {node_id: float('inf') for node_id in all_nodes_ids}
    predecessors = {node_id: None for node_id in all_nodes_ids}
    
    distances[start_node] = 0
    
    # Maintain node properties copy in memory
    node_props = {n["id"]: n["properties"] for n in initial_frame["data_state"]["structure"]["nodes"]}
    node_props[start_node]["dist"] = 0

    deltas.append({"meta": {}, "code_highlight": 2, "operations": []})
    deltas.append({"meta": {}, "code_highlight": 3, "operations": [
        {"op": "updateNodeProperties", "params": {"updates": [{"id": start_node, "properties": {"dist": 0}}]}}
    ]})

    # Phase 1: V-1 relaxation rounds
    deltas.append({"meta": {}, "code_highlight": 5, "operations": []})
    for i in range(num_nodes - 1):
        iteration_num = i + 1
        deltas.append({"meta": {"iteration": f"{iteration_num}/{num_nodes-1}", "status": "Relaxing..."}, "code_highlight": 5, "operations": []})
        
        for edge in all_edges:
            u, v, w = edge["from"], edge["to"], edge["properties"]["weight"]
            
            # >> Delta: 高亮正在松弛的边 <<
            deltas.append({
                "meta": {"iteration": f"{iteration_num}/{num_nodes-1}", "edge": f"({u}->{v}, w:{w})"}, "code_highlight": 6,
                "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": [u, v], "styleKey": "compare"}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "traversed_edge"}}
                ]
            })

            deltas.append({"meta": {"iteration": f"{iteration_num}/{num_nodes-1}", "edge": f"({u}->{v}, w:{w})"}, "code_highlight": 7, "operations": []})
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                # >> Delta: 松弛成功，更新距离和前驱 <<
                distances[v] = distances[u] + w
                predecessors[v] = u
                node_props[v]["dist"] = distances[v]
                node_props[v]["pred"] = u

                deltas.append({"meta": {"iteration": f"{iteration_num}/{num_nodes-1}", "edge": f"({u}->{v}, w:{w})"}, "code_highlight": 8, "operations": [
                    {"op": "updateNodeProperties", "params": {"updates": [{"id": v, "properties": {"dist": distances[v], "pred": u}}]}}
                ]})
            
            # >> Delta: 恢复边和节点的样式 <<
            deltas.append({"meta": {"iteration": f"{iteration_num}/{num_nodes-1}", "edge": f"({u}->{v}, w:{w})"}, "code_highlight": 6, "operations": [
                {"op": "updateNodeStyle", "params": {"ids": [u, v], "styleKey": "idle_node"}},
                {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
            ]})

    # Phase 2: Detect negative cycles
    deltas.append({"meta": {"iteration": "-", "status": "Checking for negative cycles..."}, "code_highlight": 11, "operations": []})
    for edge in all_edges:
        u, v, w = edge["from"], edge["to"], edge["properties"]["weight"]
        
        if distances[u] != float('inf') and distances[u] + w < distances[v]:
            # >> Delta: 发现负权环！<<
            # 找到环并高亮
            cycle = []
            temp_v = v
            # 防止无限循环，最多追踪V次
            for _ in range(num_nodes):
                if temp_v in cycle: # 找到环的起点
                    cycle.insert(0, temp_v)
                    break
                cycle.insert(0, temp_v)
                temp_v = predecessors.get(temp_v)
                if temp_v is None: # 路径中断
                    cycle = [] # 不是一个可追踪的环
                    break
            
            # 截取真正的环
            try:
                start_index = cycle.index(temp_v)
                cycle = cycle[start_index:]
            except (ValueError, IndexError):
                cycle = []

            cycle_edges = []
            for i in range(len(cycle) - 1):
                cycle_edges.append({"from": cycle[i], "to": cycle[i+1]})

            deltas.append({
                "meta": {"status": "Negative Cycle Detected!"}, "code_highlight": 13,
                "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": cycle, "styleKey": "swapping"}},
                    {"op": "updateEdgeStyle", "params": {"edges": cycle_edges, "styleKey": "swapping"}}
                ]
            })
            
            # 找到环后即可终止
            return { "svl_version": "5.0", "algorithm": algorithm_info, "initial_frame": initial_frame, "deltas": deltas }

    deltas.append({"meta": {"status": "Finished. No negative cycles found."}, "code_highlight": 14, "operations": []})
    
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
    # Example weighted directed graph adjacency list (can include negative weights)
    graph = {
        'A': [('B', -1), ('C', 4)],
        'B': [('C', 3), ('D', 2), ('E', 2)],
        'C': [],
        'D': [('B', 1), ('C', 5)],
        'E': [('D', -3)]
    }
    start_node = 'A'
    
    output_filename = "bellman_ford_svl_5.0.json"
    output_dir = Path("json_v5/graph")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating Bellman-Ford algorithm SVL 5.0 sequence for graph (start: {start_node})...")
    svl_output = generate_bellman_ford_svl_v5(graph, start_node)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (Bellman-Ford) sequence saved to: {output_path}")