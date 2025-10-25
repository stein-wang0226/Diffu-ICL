import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import random

def _connect_graph(G):
    """Ensures the graph is connected by adding edges between components."""
    if nx.is_connected(G):
        return G

    components = list(nx.connected_components(G))
    if len(components) < 2:
        return G

    # Add edges to connect the components
    for i in range(len(components) - 1):
        g1 = components[i]
        g2 = components[i+1]
        node1 = random.choice(list(g1))
        node2 = random.choice(list(g2))
        G.add_edge(node1, node2)
    return G

class GraphDataSampler:
    """Base class for graph data samplers."""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def sample_graph(self, batch_size):
        raise NotImplementedError

class EulerianGraphSampler(GraphDataSampler):
    """Samples a random Eulerian graph."""
    def __init__(self, num_nodes, **kwargs):
        super().__init__(num_nodes)

    def sample_graph(self, batch_size):
        """Generates a batch of random, connected, Eulerian graphs."""
        graphs = []
        for _ in range(batch_size):
            # 1. Generate a random graph with variable nodes and density
            n = random.randint(5, self.num_nodes) # Random number of nodes
            p = random.uniform(0.2, 0.8) # Random edge probability
            G = nx.erdos_renyi_graph(n, p)

            # 2. Ensure the graph is connected
            G = _connect_graph(G)

            # 3. Make the graph Eulerian
            if not nx.is_eulerian(G):
                G = nx.eulerize(G)

            # 4. Convert to torch_geometric.data.Data
            data = from_networkx(G)
            graphs.append(data)
        return graphs

def get_graph_data_sampler(data_name, num_nodes, **kwargs):
    names_to_classes = {
        "eulerian": EulerianGraphSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(num_nodes, **kwargs)
    else:
        raise NotImplementedError(f"Unknown graph sampler: {data_name}")
