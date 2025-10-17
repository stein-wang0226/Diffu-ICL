
import torch
from torch import nn

# ======== 单个 MLP Baseline ======== #
class NeuralNetwork(nn.Module):
    """
    简单 MLP Baseline，用于 ICL 对比
    接收 (xs, ys) 输入，但实际上只用 xs 预测 ys
    """
    def __init__(self, in_size, hidden_size=256, out_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, xs, ys=None, inds=None):
        """
        xs: [B, T, D]
        ys: [B, T] (unused)
        inds: optional indices
        return: [B, T]
        """
        B, T, D = xs.shape
        flat_xs = xs.view(B * T, D)          # [B*T, D]
        preds = self.net(flat_xs)            # [B*T, 1]
        preds = preds.view(B, T)             # [B, T]
        if inds is not None:
            preds = preds[:, inds]
        return preds

# ======== 多模型并行 Baseline（可选） ======== #
class ParallelNetworks(nn.Module):
    """
    多 MLP 模型并行，每个样本由一个独立的子模型处理。
    注意：仅在 batch_size == num_models 时有效。
    """
    def __init__(self, num_models, model_class, **model_class_init_args):
        super().__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for _ in range(num_models)]
        )

    def forward(self, xs, ys=None, inds=None):
        """
        xs: [B, T, D]
        """
        assert xs.shape[0] == len(self.nets), \
            f"Batch size {xs.shape[0]} != num_models {len(self.nets)}"
        B = len(self.nets)
        T = xs.shape[1]
        outs = torch.zeros(B, T, device=xs.device)
        for i, net in enumerate(self.nets):
            outs[i] = net(xs[i].unsqueeze(0), ys, inds)
        return outs
