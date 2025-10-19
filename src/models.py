import torch
from torch import nn
# Dream 如果暂时不使用，可以先注释掉
# from dllm.pipelines.dream.models.configuration_dream import DreamConfig
# from dllm.pipelines.dream.models.modeling_dream import DreamModel
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
from transformers import (
    AutoConfig, AutoModel,
    GPT2Config, GPT2Model,
    GPTJConfig, GPTJModel,
)
# ===== HuggingFace Transformers =====
from transformers import (
    AutoConfig, AutoModel,
    GPT2Config, GPT2Model,
    GPTJConfig, GPTJModel,
)

# ===== LLaMA (Llama2 / Llama3 / Llama3.1 均兼容) =====
try:
    from transformers import LlamaConfig, LlamaModel
except ImportError:
    raise ImportError("❌ 请安装 transformers>=4.40 以支持 LLaMA 模型。")
try:
    from transformers import Qwen2Config, Qwen2Model
except ImportError:
    Qwen2Config, Qwen2Model = None, None

from base_models import NeuralNetwork, ParallelNetworks
# >>> 关键：导入 LLaDA 基础模型与配置（不是 LM 包装）
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase


# ========== 工具函数：统一 x,y 拼接 ========== #
# def _combine_xs_ys(xs_b, ys_b):
#     bsize, points, dim = xs_b.shape
#     ys_b_wide = torch.cat(
#         (ys_b.view(bsize, points, 1),
#          torch.zeros(bsize, points, dim - 1, device=ys_b.device)),
#         dim=2,
#     )
#     zs = torch.stack((xs_b, ys_b_wide), dim=2).view(bsize, 2 * points, dim)
#     return zs

########## AR eval hide_last_target ########
def _combine_xs_ys(xs_b, ys_b):
    """
    Interleave (x_i, y_i) -> zs, 并把 y 扩成最后一维第一个槽位。
    xs_b: [B, T, D]
    ys_b: [B, T]
    return: zs [B, 2T, D]
    """
    bsize, points, dim = xs_b.shape
    ys_b_wide = torch.cat(
        (ys_b.view(bsize, points, 1),
         torch.zeros(bsize, points, dim - 1, device=ys_b.device, dtype=xs_b.dtype)),
        dim=2,
    )
    zs = torch.stack((xs_b, ys_b_wide), dim=2).view(bsize, 2 * points, dim)
    return zs

import torch
import torch.nn as nn
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase


import logging
# 配置日志
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)



class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128,
                    n_layer=12, n_head=4, type="gpt2", mlp_ratio=4.0,
                pretrained=False, model_name_or_path=None):
        super().__init__()
        self.family = type.lower()  # ✅ 新增
        self.mlp_ratio = mlp_ratio
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        head_dim = n_embd // n_head  # 每个注意力头的维度
        # ===== GPT2 =====
        if type == "gpt2":
            configuration = GPT2Config(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPT2Model(configuration)
        # ===== GPTJ =====
        elif type == "gptJ":
            configuration = GPTJConfig(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                rotary_dim=head_dim,       # ✅ 修复核心：rotary_dim 与 head_dim 对齐
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPTJModel(configuration)
        # ===== LLaMA 家族（支持参数自定义或预训练加载）=====
        elif self.family in ["llama", "llama2", "llama3"]:
            try:
                from transformers import LlamaConfig, LlamaModel
            except ImportError:
                raise ImportError("请安装 `transformers>=4.40` 以支持 LLaMA 模型。")

            if pretrained:
                # ✅ 直接加载预训练权重（大模型用）
                model_id = model_name_or_path or {
                    "llama3": "meta-llama/Meta-Llama-3-8B",
                    "llama2": "meta-llama/Llama-2-7b-hf",
                }.get(self.family, None)

                if model_id is None:
                    raise ValueError(f"Please provide model_name_or_path for pretrained {self.family}.")

                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                from transformers import AutoModel
                self._backbone = AutoModel.from_pretrained(model_id)
                n_embd = self._backbone.config.hidden_size

            else:
                # ✅ 自定义配置（实验版）
                if LlamaConfig is None:
                    raise ImportError("Please install transformers>=4.40 for LLaMA.")
                print(f"[Building custom {self.family.upper()} config: "
                      f"d_model={n_embd}, n_layer={n_layer}, n_head={n_head}]")
                configuration = LlamaConfig(
                    hidden_size=n_embd,
                    num_hidden_layers=n_layer,
                    num_attention_heads=n_head,
                    intermediate_size=int(n_embd * mlp_ratio),
                    max_position_embeddings=2 * n_positions,
                    # rms_norm_eps=1e-5,
                    # rope_scaling=None,
                    use_cache=False,
                )
                self._backbone = LlamaModel(configuration)


        # ===== Qwen 系列 =====
        elif self.family in ["qwen", "qwen2", "qwen2.5"]:
            if pretrained:
                model_id = model_name_or_path or {
                    "qwen": "Qwen/Qwen-7B",
                    "qwen2": "Qwen/Qwen2-7B-Instruct",
                    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
                }.get(self.family, None)
                if model_id is None:
                    raise ValueError(f"Please provide model_name_or_path for {self.family}.")
                print(f"[Loading pretrained {self.family.upper()} from {model_id}]")
                self._backbone = AutoModel.from_pretrained(model_id)
                n_embd = self._backbone.config.hidden_size
            else:
                if Qwen2Config is not None:
                    config = Qwen2Config(
                        hidden_size=n_embd,
                        num_hidden_layers=n_layer,
                        num_attention_heads=n_head,
                        intermediate_size=int(n_embd * mlp_ratio),
                        max_position_embeddings=2 * n_positions,
                        use_cache=False,
                    )
                    self._backbone = Qwen2Model(config)
                else:
                    print("⚠️ Qwen2Config not found, using AutoModel fallback.")
                    self._backbone = AutoModel.from_config(AutoConfig.from_pretrained("Qwen/Qwen2.5-7B-Instruct"))

        else:
            raise ValueError(f"Unsupported model type: {type}")


        self.name = f"{type}_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.hide_last_target = False   # ✅ 新增，用于评估时隐藏最后目标 训练不用AR有因果注意力不用隐藏
        self.predict_last_only = False  # ✅ 新增，用于评估时仅输出最后预测 + AR 单点监督

        # ===== 输入输出层 =====
        self._read_in = nn.Linear(n_dims, n_embd)
        self._read_out = nn.Linear(n_embd, 1)

        # ===== 安全维度对齐层 =====
        hidden_size = self._backbone.config.hidden_size
        self._align_proj = nn.Linear(n_embd, hidden_size) if n_embd != hidden_size else nn.Identity()

        print(f"[{type.upper()} Wrapper] n_embd={n_embd}, n_head={n_head}, "
              f"head_dim={head_dim}, rotary_dim={head_dim}, hidden_size={hidden_size}")

    def forward(self, xs, ys, inds=None):
        """
        支持 hide_last_target=True 的公平评估模式。
        xs: [B, T, D]
        ys: [B, T]
        """
        # ✅ 自动同步设备
        device = next(self.parameters()).device
        xs, ys = xs.to(device), ys.to(device)
        if inds is None:
            inds = torch.arange(ys.shape[1], device=device)
        else:
            inds = torch.as_tensor(inds, device=device)
 
        # ✅ 新增，若开启 hide_last_target 且当前在 eval 模式下，隐藏最后标签 （只影响 eval）
        if self.hide_last_target and not self.training:
            ys = ys.clone()
            ys[:,-1:] = 0.0 # # 模型在输入中看不到真实答案
            # print("ys after mask:", ys)


        zs = _combine_xs_ys(xs, ys)
        zs = zs.to(device)
        embeds = self._read_in(zs)
        embeds = self._align_proj(embeds)  # 对齐 hidden_size

        if not self.training and self.hide_last_target:
            B, T = embeds.size(0), embeds.size(1)
            ## 下三角因果掩码，确保 token_i 只能看到 <= i 的信息
            attention_mask = torch.ones((B, T), device=device, dtype=torch.float32)
            attention_mask[:, -1:] = 0  # 屏蔽最后1个
            # print("attention_mask[0]:", attention_mask[0].int().tolist())

        else:
            attention_mask = torch.ones((embeds.size(0), embeds.size(1)), device=device)

        # 手动输入 causal_mask
        try:
            outputs = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask)
        except TypeError:
            # GPT2 / GPTJ 可能不接受 attention_mask=None
            outputs = self._backbone(inputs_embeds=embeds)
        # outputs = self._backbone(inputs_embeds=embeds)
        
        
        # === 读出 ===
        h = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]   # [B, 2T, H]
        pred_all = self._read_out(h)[..., 0]   # [B, 2T]

        if self.predict_last_only:
            # 只在 x_k 位置读出（倒数第二个 token）
            return pred_all[:, -2:-1].contiguous()   # [B, 1]

        # 多点：在所有 x_i 位置读出
        return pred_all[:, ::2][:, inds]             # [B, |inds|]


# #### 原始 LLADA 双向注意力 看得到后面 tokens
# class LLaDAICLWrapper(nn.Module):
#     def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8, **extra):
#         super().__init__()
#         self.name = "llada"  # 添加 name 属性
#         # 配置 LLaDA 模型参数
#         cfg = LLaDAConfig(
#             n_heads=int(n_head),
#             n_layers=int(n_layer),
#             max_sequence_length=int(2 * n_positions),
#             rope=True,
#             alibi=False,
#             use_cache=False,
#             weight_tying=False,
#             block_group_size=int(extra.get("block_group_size", 1)),
#         )

#         # --- 关键字段统一转成干净的标量 int/float ---
#         cfg.d_model = int(n_embd)

#         # 获取 mlp_ratio（来自 YAML 或 extra 配置）
#         mlp_ratio = extra.get("mlp_ratio", getattr(cfg, "mlp_ratio", 4.0))
#         if isinstance(mlp_ratio, (list, tuple)):
#             mlp_ratio = mlp_ratio[0]
#         cfg.mlp_ratio = float(mlp_ratio)

#         # 强制设置 mlp_hidden_size 为整数
#         cfg.mlp_hidden_size = int(cfg.d_model * cfg.mlp_ratio)

#         # 设置 KV 头数（如果没有则默认使用 n_heads）
#         if not hasattr(cfg, "effective_n_kv_heads") or cfg.effective_n_kv_heads is None:
#             cfg.effective_n_kv_heads = int(n_head)

#         # 打印模型配置（用于调试）
#         print(f"[LLaDA Wrapper] d_model={cfg.d_model}, mlp_ratio={cfg.mlp_ratio}, "
#               f"mlp_hidden_size={cfg.mlp_hidden_size}, n_heads={cfg.n_heads}, "
#               f"n_layers={cfg.n_layers}, kv_heads={cfg.effective_n_kv_heads}, "
#               f"block_group_size={cfg.block_group_size}")

#         # LLaDA 模型 backbone
#         self._backbone = _LLaDABase(cfg, init_params=True)

#         # 与 GPT 对齐的输入/输出层
#         self.n_positions = n_positions
#         self.n_dims = n_dims
#         self.d_model = cfg.d_model
#         self._read_in = nn.Linear(n_dims, cfg.d_model)
#         self._read_out = nn.Linear(cfg.d_model, 1)

#     def forward(self, xs, ys, inds=None):
#         # 获取 batch_size 和 seq_len
#         b, t, d = xs.shape

#         # 默认设置 inds
#         if inds is None:
#             inds = torch.arange(ys.shape[1], device=ys.device)
#         else:
#             inds = torch.as_tensor(inds, device=ys.device)

#         # 将 xs 和 ys 交错并合并
#         ys_wide = torch.cat([ys.view(b, t, 1), torch.zeros(b, t, d - 1, device=ys.device)], dim=2)
#         zs = torch.stack([xs, ys_wide], dim=2).view(b, 2 * t, d)

#         # 输入 LLaDA 模型
#         input_ids = zs  # 使用 zs 作为 input_ids 传递给模型
#         embeds = self._read_in(input_ids)  # 嵌入映射到模型的维度

#         # 确保 embeds 维度正确 (b, seq_len, d_model)
#         embeds = embeds.view(b, 2 * t, self.d_model)

#         # 调用 _backbone 进行前向传播
#         out = self._backbone(input_ids=input_ids, input_embeddings=embeds, output_hidden_states=True)  # 确保传递 input_ids
#         last_h = out.hidden_states[-1]

#         # 通过输出层获取预测
#         pred = self._read_out(last_h)[:, ::2, 0][:, inds]

#         return pred


#### hide_last_y #########
class LLaDAICLWrapper(nn.Module):
    """
    公平版 LLaDA Wrapper（非自回归）：
    - 不使用因果掩码（因为 diffusion 本身非自回归）
    - 使用 hide_last_target=True 避免输入泄漏 (防止模型看到 y_k)
    - 支持 predict_last_only=True，只返回最后一步预测 (few-shot ICL)
    """
    def __init__(
        self,
        n_dims,
        n_positions,
        n_embd=256,
        n_layer=12,
        n_head=8,
        *,
        mlp_ratio=4.0,
        block_group_size=1,
        hide_last_target=True,   # ✅ 屏蔽最后一个 y_k 的输入
        predict_last_only=True,  # ✅ 只评估最后一个目标（few-shot 模式）
        **extra,
    ):
        super().__init__()
        self.name = "llada"
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.hide_last_target = bool(hide_last_target)
        self.predict_last_only = bool(predict_last_only)

        # ===== 配置 LLaDA =====
        cfg = LLaDAConfig(
            n_heads=int(n_head),
            n_layers=int(n_layer),
            max_sequence_length=int(2 * n_positions),
            rope=True,
            alibi=False,
            use_cache=False,
            weight_tying=False,
            block_group_size=int(block_group_size),
        )
        cfg.d_model = int(n_embd)
        cfg.mlp_ratio = float(mlp_ratio)
        cfg.mlp_hidden_size = int(cfg.d_model * cfg.mlp_ratio)
        cfg.effective_n_kv_heads = int(getattr(cfg, "effective_n_kv_heads", n_head))

        print(
            f"[LLaDA Wrapper] d_model={cfg.d_model}, mlp_ratio={cfg.mlp_ratio}, "
            f"mlp_hidden_size={cfg.mlp_hidden_size}, n_heads={cfg.n_heads}, "
            f"n_layers={cfg.n_layers}, kv_heads={cfg.effective_n_kv_heads}, "
            f"block_group_size={cfg.block_group_size}, "
            f"hide_last_target={self.hide_last_target}, predict_last_only={self.predict_last_only}"
        )

        # ===== Backbone =====
        self._backbone = _LLaDABase(cfg, init_params=True)

        # ===== 与 GPT 对齐的投/读层 =====
        self.d_model = cfg.d_model
        self._read_in = nn.Linear(n_dims, cfg.d_model)
        self._read_out = nn.Linear(cfg.d_model, 1)

    def forward(self, xs, ys, inds=None):
        """
        xs: [B, T, D]
        ys: [B, T]
        逻辑：
        - 屏蔽最后一个 y_k（输入 0）
        - 输入为 [x1,y1,...,xk,0]
        - 输出取 x 位置的预测 [B, T]
        - 若 predict_last_only=True，只返回最后一个预测
        """
        b, t, d = xs.shape

        # ===== 处理 inds（默认评估所有步或最后一步）=====
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.as_tensor(inds, device=ys.device)

        # ===== 屏蔽最后一个 y_k 输入 =====
        if self.hide_last_target:
            ys_for_input = ys.clone()
            ys_for_input[:, -1] = 0.0
        else:
            ys_for_input = ys

        # ===== 构造交错输入 [x1,y1,x2,y2,...,xT,yT] =====
        ys_wide = torch.cat(
            [ys_for_input.view(b, t, 1),
             torch.zeros(b, t, d - 1, device=ys.device, dtype=xs.dtype)],
            dim=2,
        )  # [B,T,D]
        zs = torch.stack([xs, ys_wide], dim=2).view(b, 2 * t, d)  # [B,2T,D]

        # ===== 输入投影 =====
        embeds = self._read_in(zs).view(b, 2 * t, self.d_model)  # [B,2T,d_model]

        # ===== 前向传播 =====
        out = self._backbone(
            input_ids=zs,                    # 传入原输入
            input_embeddings=embeds,         # 明确传入 embedding
            output_hidden_states=True,
        )
        last_h = out.hidden_states[-1]  # [B,2T,d_model]

        # ===== 在 x 位置读出预测（::2 对齐）=====
        pred_all = self._read_out(last_h)[:, ::2, 0]  # [B, T]

        # ===== 返回结果 =====
        if self.predict_last_only:
            return pred_all[:, -1:].contiguous()  # [B,1]
        else:
            return pred_all[:, inds].contiguous()



# ========== 构建函数 ========== #
def build_model(conf):
    family = conf["family"]

    # 🧭 兼容不同命名 （s）
    n_layer = conf.get("n_layer", conf.get("n_layers", 6))
    n_head = conf.get("n_head", conf.get("n_heads", 8))
    n_embd = conf.get("n_embd", conf.get("d_model", 256))


    if family in ["gpt2", "gptJ", "gptj", "qwen", "qwen2", "qwen2.5", "llama", "llama2", "llama3"]:
        model = TransformerModel(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type=family,
            pretrained=conf.get("pretrained", False),
            model_name_or_path=conf.get("model_name_or_path", None),
        )

        # ✅ 从配置文件中读取公平性控制参数
        model.hide_last_target = conf.get("hide_last_target", False)
        model.predict_last_only = conf.get("predict_last_only", False)

        print(f"[build_model] Initialized {family} model "
              f"(hide_last_target={model.hide_last_target}, predict_last_only={model.predict_last_only})")
        return model
    elif family == "llada":
        return LLaDAICLWrapper(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            mlp_ratio=conf.get("mlp_ratio", 4.0), # 这个参数是否要统一
        )
    else:
        raise NotImplementedError(f"Unsupported model family: {family}")






def get_relevant_baselines(task_name):
    # 将任务名称映射到对应baseline模型列表
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):# xs：[batch_size, n_points, n_dims]  , ys:[batch_size, n_points]
        # 返回形状为 [batch_size, len(inds)] 的张量，包含指定位置的预测结果。
        if inds is None:
            inds = range(ys.shape[1]) # 默认 [0,n]
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point # 第一个点的预测值为 0 ,没有可供参考的历史点，预测值直接设为 0
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i] # train_xs 和 train_ys：从输入中提取历史点的特征 , 标签
            test_x = xs[:, i : i + 1] # 当前测试i点的特征
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()# 当前测试点与所有历史点之间的欧几里得距离

            if self.weights == "uniform":  # 权重相同
                weights = torch.ones_like(dist) # 权重与距离成反比
            else:
                weights = 1.0 / dist #
                inf_mask = torch.isinf(weights).float()  # deal with exact match # 处理距离为零的情况（feature is same）
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row] # 1

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]  # 选择 topk
            for y, w, n in zip(train_ys, weights, ranks): # n:topk的切片索引
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum()) # topk 求加权平均
            preds.append(torch.stack(pred))
        # 将所有位置的预测结果拼接成一个张量，形状为 [batch_size, len(inds)]
        return torch.stack(preds, dim=1) # 讲inds维拼接


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank
# due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver # torch.linalg.lstsq 中的求解器（driver）
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu() # 数据移动到 CPU
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            ) # 根据train x,y 求解得到 线形的w matrix
            pred = test_x @ ws # 利用w得到预测值 [batch_size, 1, 1]
            preds.append(pred[:, 0, 0]) # [bs]

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1) # 直接计算 w
            pred = test_x @ w_p # pred
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]): # 每个prompt分别计算预测值
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1) #

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)




# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i] #
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
