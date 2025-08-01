import json
from pathlib import Path
from collections import deque
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

def generate_bfs_svl_v5(graph_adj_list: dict, start_node: str):
    """
    Generate complete visualization sequence for Breadth-First Search (BFS) according to SVL 5.0 specification.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Breadth-First Search (BFS)",
        "family": "Graph"
    }

    variables_schema = [
        {"name": "currentNode", "type": "value", "description": "Currently dequeued node"},
        {"name": "queue", "type": "list", "description": "Queue used for BFS"}
    ]

    data_schema = {
        "node_properties_schema": [
            {"name": "distance", "type": "value", "description": "Distance from start node"},
            {"name": "visited", "type": "flag", "description": "Whether already visited"}
        ]
    }
    
    pseudocode = [
        "function BFS(graph, startNode):",
        "  queue = new Queue()",
        "  visited = new Set()",
        "  queue.enqueue(startNode)",
        "  visited.add(startNode)",
        "  while queue is not empty:",
        "    currentNode = queue.dequeue()",
        "    // process(currentNode)",
        "    for neighbor in graph.getNeighbors(currentNode):",
        "      if neighbor is not visited:",
        "        visited.add(neighbor)",
        "        queue.enqueue(neighbor)",
    ]

    # 2. Build initial frame
    nodes = []
    for node_id in graph_adj_list.keys():
        nodes.append({
            "id": node_id,
            "styleKey": "idle_node",
            "properties": {
                "distance": "inf",
                "visited": False
            }
        })

    edges = []
    edge_set = set()
    for u, neighbors in graph_adj_list.items():
        for v in neighbors:
            if tuple(sorted((u, v))) not in edge_set:
                edges.append({"from": u, "to": v, "directed": False, "styleKey": "normal_edge"})
                edge_set.add(tuple(sorted((u, v))))

    initial_frame = {
        "data_schema": data_schema,
        "data_state": {
            "type": "graph",
            "structure": { "nodes": nodes, "edges": edges }
        },
        "variables_schema": variables_schema,
        "pseudocode": pseudocode,
        "code_highlight": 1,
        "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    queue = deque([start_node])
    visited = {start_node}
    
    # Initialize start node
    start_node_obj = next(n for n in initial_frame["data_state"]["structure"]["nodes"] if n["id"] == start_node)
    start_node_obj["properties"]["distance"] = 0
    start_node_obj["properties"]["visited"] = True
    start_node_obj["styleKey"] = "current_node"

    deltas.append({"meta": {"queue": list(queue)}, "code_highlight": 4, "operations": [
        {"op": "updateNodeProperties", "params": {"updates": [{"id": start_node, "properties": {"distance": 0, "visited": True}}]}},
        {"op": "updateNodeStyle", "params": {"ids": [start_node], "styleKey": "current_node"}}
    ]})
    deltas.append({"meta": {"queue": list(queue)}, "code_highlight": 5, "operations": []})

    while queue:
        deltas.append({"meta": {"queue": list(queue)}, "code_highlight": 6, "operations": []})
        u = queue.popleft()
        
        # Node dequeued, becomes current node
        deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 7, "operations": [
            {"op": "updateNodeStyle", "params": {"ids": [u], "styleKey": "current_node"}}
        ]})
        
        deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 9, "operations": []})
        
        u_dist = next(n["properties"]["distance"] for n in initial_frame["data_state"]["structure"]["nodes"] if n["id"] == u)

        for v in graph_adj_list[u]:
            deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 10, "operations": [
                 {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "traversed_edge"}}
            ]})

            if v not in visited:
                visited.add(v)
                queue.append(v)
                
                # Discover unvisited neighbor, update properties and enqueue
                deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 11, "operations": [
                    {"op": "updateNodeProperties", "params": {"updates": [{"id": v, "properties": {"distance": u_dist + 1, "visited": True}}]}}
                ]})
                node_obj = next(n for n in initial_frame["data_state"]["structure"]["nodes"] if n["id"] == v)
                node_obj["properties"]["distance"] = u_dist + 1
                node_obj["properties"]["visited"] = True
                deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 12, "operations": [
                    {"op": "updateNodeStyle", "params": {"ids": [v], "styleKey": "compare"}},
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
                ]})
            else:
                 deltas.append({"meta": {"currentNode": u, "queue": list(queue)}, "code_highlight": 10, "operations": [
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
                 ]})

        # Current node processed, mark as visited
        deltas.append({"meta": {"currentNode": "-", "queue": list(queue)}, "code_highlight": 6, "operations": [
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
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    start_node = 'A'
    
    output_filename = "bfs_svl_5.0.json"
    output_dir = Path("json_v5/graph") 
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating BFS SVL 5.0 sequence for graph (start: {start_node})...")
    svl_output = generate_bfs_svl_v5(graph, start_node)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (BFS) sequence saved to: {output_path}")