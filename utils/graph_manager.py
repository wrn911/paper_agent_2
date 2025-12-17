# utils/graph_manager.py
# This file contains the GraphManager class for interacting with the knowledge graph.

import json
import networkx as nx
from networkx.readwrite import json_graph

class GraphManager:
    """
    Manages loading and querying the knowledge graph stored in a JSON format.
    """
    def __init__(self, graph_path: str):
        """
        Initializes the GraphManager by loading the graph from the specified path.

        Args:
            graph_path (str): The path to the JSON graph file.
        """
        self.graph = self._load_graph(graph_path)

    def _load_graph(self, path: str) -> nx.Graph:
        """
        Loads a graph from a JSON file into a NetworkX graph object.
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return json_graph.node_link_graph(data, edges="edges") 
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading graph from {path}: {e}")
            # Return an empty graph on error
            return nx.Graph()

    def _extract_relation_text(self, edge_data: dict) -> str:
        """
        Extracts the relation_text from edge attributes, handling both Graph and MultiGraph.
        """
        if not isinstance(edge_data, dict):
            return ""
        if self.graph.is_multigraph():
            for attrs in edge_data.values():
                if isinstance(attrs, dict) and 'relation_text' in attrs:
                    return attrs.get('relation_text', '')
            return ""
        return edge_data.get('relation_text', '')

    def get_nodes(self, node_type: str, task: str) -> list:
        """
        Returns a list of node IDs that match the specified type and task.

        Args:
            node_type (str): The type of the node (e.g., 'train', 'test').
            task (str): The task the node is associated with (e.g., 'reproduction').
        
        Returns:
            list: A list of matching node IDs.
        """
        matching_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_tasks = data.get('task')
            if not isinstance(node_tasks, list): # Ensure node_tasks is a list for consistent checking
                node_tasks = [node_tasks] if node_tasks else []

            if data.get('node_type') == node_type and task in node_tasks:
                matching_nodes.append(node_id)
        return matching_nodes

    def get_neighbors_context(self, node_id: str) -> dict:
        """
        Finds the neighbors of the current node and constructs a context dictionary
        for the prompt.
        """
        if node_id not in self.graph:
            return {}

        context = {}
        # This is a simplified logic. A real implementation would handle multiple neighbors.
        i = 1
        for neighbor in self.graph.neighbors(node_id):
            edge_data = self.graph.get_edge_data(node_id, neighbor)
            neighbor_data = self.graph.nodes[neighbor]

            context[f'model{i}'] = neighbor
            context[f'idea{i}'] = neighbor_data.get('idea', '')
            context[f'relation{i}'] = self._extract_relation_text(edge_data)
            # Placeholder for adding example code and config
            context[f'code{i}'] = neighbor_data.get('model_code', '')
            context[f'config{i}'] = neighbor_data.get('config_yaml', '')
            
            i += 1
            if i > 3: # Limit to 3 neighbors for the example
                break

        return context
