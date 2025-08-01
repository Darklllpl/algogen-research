import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from final_styles import DEFAULT_STYLES

# Disjoint Set Union (DSU) helper class
class DSU:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def generate_kruskal_svl_v5(graph_def: dict):
    """
    Generate complete visualization sequence for Kruskal algorithm according to SVL 5.0 specification.
    This version uses auxiliary_views to clearly display DSU state changes.
    """
    
    # 1. Define static components
    algorithm_info = {
        "name": "Kruskal's Algorithm",
        "family": "Graph MST"
    }

    variables_schema = [
        {"name": "currentEdge", "type": "value", "description": "Current edge being considered"},
        {"name": "mst_cost", "type": "value", "description": "Current MST total weight"}
    ]

    data_schema = {} # Kruskal doesn't track internal properties
    
    pseudocode = [
        "function Kruskal(graph):",
        "  MST = []",
        "  all_edges = graph.get_all_edges()",
        "  all_edges.sort_by_weight()",
        "  dsu = new DisjointSetUnion(graph.nodes)",
        "  for each edge (u, v, w) in all_edges:",
        "    if dsu.find(u) != dsu.find(v):",
        "      MST.add((u, v, w))",
        "      dsu.union(u, v)",
        "  return MST",
    ]

    # 2. Build initial frame
    all_nodes_ids = list(graph_def.keys())
    nodes = [{"id": n, "styleKey": "idle_node", "properties": {}} for n in all_nodes_ids]
    
    edges_list, edge_set = [], set()
    for u, neighbors in graph_def.items():
        for v, weight in neighbors:
            if tuple(sorted((u, v))) not in edge_set:
                edges_list.append({"u": u, "v": v, "weight": weight})
                edge_set.add(tuple(sorted((u, v))))

    # Sort all edges by weight
    sorted_edges = sorted(edges_list, key=lambda x: x['weight'])

    # Initialize DSU auxiliary view
    node_to_idx = {node_id: i for i, node_id in enumerate(all_nodes_ids)}
    initial_dsu_parents = [node_id for node_id in all_nodes_ids]
    
    dsu_view = {
        "view_id": "dsu_view",
        "type": "table",
        "title": "Disjoint Set Union (Parent Array)",
        "data": [initial_dsu_parents],
        "options": {
            "row_headers": ["parent"],
            "col_headers": all_nodes_ids
        }
    }

    initial_frame = {
        "data_schema": data_schema,
        "data_state": { "type": "graph", "structure": { 
            "nodes": nodes, 
            "edges": [{"from": e["u"], "to": e["v"], "directed": False, "label": str(e["weight"]), "styleKey": "normal_edge"} for e in sorted_edges]
        }},
        "auxiliary_views": [dsu_view],
        "variables_schema": variables_schema, "pseudocode": pseudocode,
        "code_highlight": 1, "styles": DEFAULT_STYLES
    }

    # 3. Track algorithm execution, generate Deltas
    deltas = []
    dsu = DSU(all_nodes_ids)
    mst_cost = 0
    
    deltas.append({"meta": {}, "code_highlight": 4, "operations": []})
    deltas.append({"meta": {}, "code_highlight": 5, "operations": []})
    deltas.append({"meta": {}, "code_highlight": 6, "operations": []})

    for edge in sorted_edges:
        u, v, w = edge["u"], edge["v"], edge["weight"]
        
        # >> Delta: Consider next minimum weight edge <<
        deltas.append({
            "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 7,
            "operations": [{"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "compare"}}]
        })
        
        root_u = dsu.find(u)
        root_v = dsu.find(v)

        # >> Delta: Check if both endpoints of this edge are in the same set <<
        deltas.append({
            "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 7,
            "operations": [
                {"op": "highlightTableCell", "params": {"view_id": "dsu_view", "cells": [{"row":0, "col":node_to_idx[u]}, {"row":0, "col":node_to_idx[v]}], "styleKey": "compare"}}
            ]
        })
        
        if root_u != root_v:
            # >> Delta: Not forming a cycle, accept this edge <<
            dsu.union(u, v)
            mst_cost += w
            
            # Get updated parent array
            updated_dsu_parents = [dsu.parent[node_id] for node_id in all_nodes_ids]
            
            deltas.append({
                "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 8,
                "operations": [{"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "in_path_edge"}}]
            })
            
            # >> Delta: Update DSU view <<
            deltas.append({
                "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 9,
                "operations": [
                    {"op": "updateTableCell", "params": {"view_id": "dsu_view", "updates": 
                        [{"row": 0, "col": node_to_idx[node], "value": parent} for node, parent in dsu.parent.items()]
                    }}
                ]
            })

        else:
            # >> Delta: Forming a cycle, reject this edge <<
            deltas.append({
                "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 7,
                "operations": [
                    # Use swapping style (usually red) to indicate rejection
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "swapping"}}
                ]
            })
            deltas.append({
                "meta": {"currentEdge": f"({u}-{v}, w:{w})", "mst_cost": mst_cost}, "code_highlight": 7,
                "operations": [
                    # Restore normal edge style
                    {"op": "updateEdgeStyle", "params": {"edges": [{"from": u, "to": v}], "styleKey": "normal_edge"}}
                ]
            })


    deltas.append({"meta": {"mst_cost": mst_cost}, "code_highlight": 10, "operations": []})

    # 4. Assemble and return final SVL object
    final_svl_object = {
        "svl_version": "5.0",
        "algorithm": algorithm_info,
        "initial_frame": initial_frame,
        "deltas": deltas
    }
    
    return final_svl_object

# --- Example usage ---
if __name__ == '__main__':
    graph = {
        'A': [('B', 2), ('D', 5)],
        'B': [('A', 2), ('C', 4), ('D', 1), ('E', 3)],
        'C': [('B', 4), ('E', 6)],
        'D': [('A', 5), ('B', 1), ('E', 5)],
        'E': [('B', 3), ('C', 6), ('D', 5)]
    }
    
    output_filename = "kruskal_svl_5.0.json"
    output_dir = Path("json_v5/graph")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_filename

    print(f"Generating Kruskal algorithm SVL 5.0 sequence for graph...")
    svl_output = generate_kruskal_svl_v5(graph)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(svl_output, f, indent=2, ensure_ascii=False) 

    print(f"SVL 5.0 (Kruskal) sequence successfully saved to file: {output_path}")