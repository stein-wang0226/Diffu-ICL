import math

import torch
import random
import numpy as np

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

def rand_select_sampler(sampler1, sampler2): # todo 添加随机选择sample
    if random.random() < 0.5:
        return sampler1
    else:
        return sampler2


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform":UniformSampler
        # add
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError



def sample_transformation(eigenvalues, normalize=False): # 根据给定的特征值（eigenvalues）生成一个线性变换矩阵，用于对数据进行线性变换
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale
    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None: #
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else: # 利用seeds
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
                # 缩放矩阵
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b


class UniformSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            a = -1.96 #
            b=   1.96
            xs_b = a + (b - a) * torch.rand(b_size, n_points, self.n_dims)

            # xs_b = torch.rand(b_size, n_points, self.n_dims) # U [a,b]  [-+1.96]
        else: # 利用seed
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                np.random.seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None: # 截断特征维度 将多余的维度置零
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

