import json
from pathlib import Path
from collections import deque
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_topological_sort_svl_v5(graph_adj_list: dict):
    """
    Generate complete visualization sequence for Topological Sort (Kahn's algorithm) according to SVL 5.0 specification.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Topological Sort (Kahn's Algorithm)",
        "family": "Graph"
    }

    variables_schema = [
        {"name": "queue", "type": "list", "description": "Queue of nodes with in-degree 0"},
        {"name": "sortedResult", "type": "list", "description": "Topological sort result"},
        {"name": "status", "type": "value", "description": "Algorithm final status"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "in_degree", "type": "value", "description": "Node in-degree"}
        ]
    }
    
    pseudocode = [
        "function TopologicalSort(graph):",
        "  in_degree = compute_in_degrees(graph)",
        "  queue = new Queue(all nodes with in_degree 0)",
        "  sorted_result = []",
        "  while queue is not empty:",
        "    u = queue.dequeue()",
        "    sorted_result.append(u)",
        "    for each neighbor v of u:",
        "      in_degree[v] = in_degree[v] - 1",
        "      if in_degree[v] == 0:",
        "        queue.enqueue(v)",
        "  if len(sorted_result) != num_nodes:",
        "    return 'Graph has a cycle!'",
        "  return sorted_result",
    ]

    # 2. Build initial frame
    all_nodes_ids = list(graph_adj_list.keys())
    
    # Calculate initial in-degrees
    in_degrees = {node_id: 0 for node_id in all_nodes_ids}
    for u, neighbors in graph_adj_list.items():
        for v in neighbors:
            if v in in_degrees:
                in_degrees[v] += 1

    nodes = [{"id": n, "styleKey": "idle_node", "properties": {"in_degree": in_degrees[n]}} for n in all_nodes_ids]
    
    edges = []
    for u, neighbors in graph_adj_list.items():
        for v in neighbors:
            # Topological sort handles directed graphs
            edges.append({"from": u, "to": v, "directed": True, "styleKey": "normal_edge"})

    initial_frame = {
        "data_schema": data_schema,
        "data_state": {"type": "graph", "structure": {"nodes": nodes, "edges": edges}},
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    
    # Initialize algorithm data structures
    # Note: in_degrees already calculated above, use directly
    queue = deque([node for node, degree in in_degrees.items() if degree == 0])
    sorted_result = []
    
    # Maintain node properties copy in memory
    node_props = {n["id"]: n["properties"] for n in initial_frame["data_state"]["structure"]["nodes"]}
    
    deltas.append({"meta": {}, "code_highlight": 2, "operations": []})
    deltas.append({"meta": {"queue": list(queue)}, "code_highlight": 3, "operations": [
        {"op": "updateNodeStyle", "params": {"ids": list(queue), "styleKey": "compare"}}
    ]})
    deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 4, "operations": []})

    while queue:
        deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 5, "operations": []})
        u = queue.popleft()
        
        # >> Delta: 节点出队，加入结果列表 <<
        sorted_result.append(u)
        deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 6, "operations": [
            {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "current_node"}}
        ]})
        deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 7, "operations": [
            {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "sorted"}}
        ]})

        deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 8, "operations": []})
        for v in graph_adj_list.get(u, []):
            # >> Delta: 处理出边，减少邻居的入度 <<
            # 使用红色（swapping样式）临时高亮被“移除”的边
            deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 9, "operations": [
                {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "swapping"}}
            ]})
            
            in_degrees[v] -= 1
            node_props[v]["in_degree"] = in_degrees[v]
            deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 9, "operations": [
                {"op": "updateNodeProperties", "params": {"updates": [{"id": v, "properties": {"in_degree": in_degrees[v]}}]}},
                {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
            ]})

            if in_degrees[v] == 0:
                # >> Delta: 邻居入度变为0，将其入队 <<
                queue.append(v)
                deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 10, "operations": []})
                deltas.append({"meta": {"queue": list(queue), "sortedResult": sorted_result}, "code_highlight": 11, "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "compare"}}
                ]})
    
    # >> Delta: 检查图中是否存在环路 <<
    if len(sorted_result) != len(all_nodes_ids):
        deltas.append({"meta": {"status": "Graph has a cycle!"}, "code_highlight": 13, "operations": []})
    else:
        deltas.append({"meta": {"status": "Topological sort successful!"}, "code_highlight": 14, "operations": []})

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
    # Example directed acyclic graph (DAG) adjacency list
    graph = {
        'A': ['C'],
        'B': ['C', 'D'],
        'C': ['E'],
        'D': ['F'],
        'E': ['F'],
        'F': []
    }
    
    output_filename = "topological_sort_svl_5.0.json"
    output_dir = Path("json_v5/graph") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating Topological Sort SVL 5.0 sequence for graph...")
    svl_output = generate_topological_sort_svl_v5(graph)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (Topological Sort) sequence saved to: {output_path}")