import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_dfs_svl_v5(graph_adj_list: dict, start_node: str):
    """
    Generate complete visualization sequence for Depth-First Search (DFS) according to SVL 5.0 specification.
    This version uses iterative approach and tracks node discovery and finish times.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Depth-First Search (DFS)",
        "family": "Graph"
    }

    variables_schema = [
        {"name": "currentNode", "type": "value", "description": "Current node at top of stack"},
        {"name": "stack", "type": "list", "description": "Stack used for DFS"},
        {"name": "time", "type": "value", "description": "Global timestamp"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "d", "type": "value", "description": "Discovery time"},
            {"name": "f", "type": "value", "description": "Finish time"},
            {"name": "visited", "type": "flag", "description": "Whether already visited"}
        ]
    }
    
    pseudocode = [
        "function DFS_Iterative(graph, startNode):",
        "  stack = new Stack()",
        "  visited = new Set()",
        "  time = 0",
        "  stack.push(startNode)",
        "  while stack is not empty:",
        "    u = stack.peek()",
        "    if u is not visited:",
        "      visited.add(u); time++",
        "      discovery_time[u] = time",
        "    found_new = false",
        "    for v in graph.getNeighbors(u):",
        "      if v is not visited:",
        "        stack.push(v)",
        "        found_new = true; break",
        "    if not found_new:",
        "      stack.pop()",
        "      time++",
        "      finish_time[u] = time",
    ]

    # 2. Build initial frame
    nodes = []
    for node_id in graph_adj_list.keys():
        nodes.append({
            "id": node_id, "styleKey": "idle_node",
            "properties": {"d": "-", "f": "-", "visited": False}
        })

    edges, edge_set = [], set()
    for u, neighbors in graph_adj_list.items():
        for v in neighbors:
            if tuple(sorted((u, v))) not in edge_set:
                edges.append({"from": u, "to": v, "directed": False, "styleKey": "normal_edge"})
                edge_set.add(tuple(sorted((u, v))))

    initial_frame = {
        "data_schema": data_schema,
        "data_state": { "type": "graph", "structure": { "nodes": nodes, "edges": edges } },
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    stack = [start_node]
    visited = set()
    time = 0
    
    # Maintain node properties copy in memory for easy access
    node_props = {n["id"]: n["properties"] for n in initial_frame["data_state"]["structure"]["nodes"]}

    deltas.append({"meta": {"stack": stack, "time": time}, "code_highlight": 5, "operations": []})

    while stack:
        u = stack[-1] # Peek
        
        deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 7, "operations": [
            {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "current_node"}}
        ]})

        if u not in visited:
            visited.add(u)
            time += 1
            node_props[u]["d"] = time
            node_props[u]["visited"] = True

            # First visit to node, record discovery time
            deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 9, "operations": [
                {"op": "updateNodeProperties", "params": {"updates": [{"id": u, "properties": {"d": time, "visited": True}}]}}
            ]})
            deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 10, "operations": []})

        found_new = False
        deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 12, "operations": []})
        for v in graph_adj_list[u]:
            if v not in visited:
                # Found unvisited neighbor, highlight edge and push to stack
                deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 13, "operations": [
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "traversed_edge"}}
                ]})
                stack.append(v)
                found_new = True
                deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 14, "operations": [
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}},
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "compare"}}
                ]})
                deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 15, "operations": []})
                break
        
        if not found_new:
            stack.pop()
            time += 1
            node_props[u]["f"] = time
            
            # No unvisited neighbors, node finished, record finish time and pop
            deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 17, "operations": []})
            deltas.append({"meta": {"stack": stack, "time": time, "currentNode": u}, "code_highlight": 19, "operations": [
                {"op": "updateNodeProperties", "params": {"updates": [{"id": u, "properties": {"f": time}}]}},
                {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "visited_node"}}
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
    graph = {
        'A': ['B', 'C'], 'B': ['A', 'D', 'E'], 'C': ['A', 'F'],
        'D': ['B'], 'E': ['B', 'F'], 'F': ['C', 'E']
    }
    start_node = 'A'
    
    output_filename = "dfs_svl_5.0.json"
    output_dir = Path("json_v5/graph") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating DFS SVL 5.0 sequence for graph (start: {start_node})...")
    svl_output = generate_dfs_svl_v5(graph, start_node)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (DFS) sequence saved to: {output_path}")