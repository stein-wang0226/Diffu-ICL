import torch
import torch_geometric

def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()

class GraphTask:
    def __init__(self, batch_size):
        self.b_size = batch_size

    def evaluate(self, graphs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError

class MaxDegreePrediction(GraphTask):
    def __init__(self, batch_size, **kwargs):
        super().__init__(batch_size)

    def evaluate(self, graphs):
        max_degrees = []
        for graph in graphs:
            if graph.num_edges == 0:
                max_degrees.append(0.0)
                continue
            degrees = torch_geometric.utils.degree(graph.edge_index[0], num_nodes=graph.num_nodes)
            max_degrees.append(torch.max(degrees).float())
        return torch.tensor(max_degrees)

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

def get_graph_task_sampler(task_name, batch_size, **kwargs):
    task_names_to_classes = {
        "max_degree_prediction": MaxDegreePrediction,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        return lambda **args: task_cls(batch_size, **args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown graph task: {task_name}")
