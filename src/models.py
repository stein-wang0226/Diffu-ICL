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
from transformers import GPT2Model, GPT2Config, GPTJModel, GPTJConfig
from base_models import NeuralNetwork, ParallelNetworks
# >>> 关键：导入 LLaDA 基础模型与配置（不是 LM 包装）
from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.pipelines.llada.models.modeling_llada import LLaDAModel as _LLaDABase


# ========== 工具函数：统一 x,y 拼接 ========== #
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
         torch.zeros(bsize, points, dim - 1, device=ys_b.device)),
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


# ========== GPT 模型 ========== #
class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, type="gpt2", mlp_ratio=4.0):
        super().__init__()
        self.mlp_ratio = mlp_ratio  # 添加 mlp_ratio 参数

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
        elif type == "gptJ":
            configuration = GPTJConfig(
                n_positions=2 * n_positions,
                n_embd=n_embd,
                n_layer=n_layer,
                n_head=n_head,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
                use_cache=False,
            )
            self._backbone = GPTJModel(configuration)
        else:
            raise ValueError(f"Unsupported GPT type: {type}")

        self.name = f"{type}_embd={n_embd}_layer={n_layer}_head={n_head}"
        
        self.n_positions = n_positions
        self.n_dims = n_dims

        self._read_in = nn.Linear(n_dims, n_embd)
        self._read_out = nn.Linear(n_embd, 1)
        # self._read_out = nn.Linear(int(n_embd * mlp_ratio), 1)  # 使用 mlp_ratio 进行调整

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.as_tensor(inds, device=ys.device)

        zs = _combine_xs_ys(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]


class LLaDAICLWrapper(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=256, n_layer=12, n_head=8, **extra):
        super().__init__()
        self.name = "llada"  # 添加 name 属性
        # 配置 LLaDA 模型参数
        cfg = LLaDAConfig(
            n_heads=int(n_head),
            n_layers=int(n_layer),
            max_sequence_length=int(2 * n_positions),
            rope=True,
            alibi=False,
            use_cache=False,
            weight_tying=False,
            block_group_size=int(extra.get("block_group_size", 1)),
        )

        # --- 关键字段统一转成干净的标量 int/float ---
        cfg.d_model = int(n_embd)

        # 获取 mlp_ratio（来自 YAML 或 extra 配置）
        mlp_ratio = extra.get("mlp_ratio", getattr(cfg, "mlp_ratio", 4.0))
        if isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = mlp_ratio[0]
        cfg.mlp_ratio = float(mlp_ratio)

        # 强制设置 mlp_hidden_size 为整数
        cfg.mlp_hidden_size = int(cfg.d_model * cfg.mlp_ratio)

        # 设置 KV 头数（如果没有则默认使用 n_heads）
        if not hasattr(cfg, "effective_n_kv_heads") or cfg.effective_n_kv_heads is None:
            cfg.effective_n_kv_heads = int(n_head)

        # 打印模型配置（用于调试）
        print(f"[LLaDA Wrapper] d_model={cfg.d_model}, mlp_ratio={cfg.mlp_ratio}, "
              f"mlp_hidden_size={cfg.mlp_hidden_size}, n_heads={cfg.n_heads}, "
              f"n_layers={cfg.n_layers}, kv_heads={cfg.effective_n_kv_heads}, "
              f"block_group_size={cfg.block_group_size}")

        # LLaDA 模型 backbone
        self._backbone = _LLaDABase(cfg, init_params=True)

        # 与 GPT 对齐的输入/输出层
        self.n_positions = n_positions
        self.n_dims = n_dims
        self.d_model = cfg.d_model
        self._read_in = nn.Linear(n_dims, cfg.d_model)
        self._read_out = nn.Linear(cfg.d_model, 1)

    def forward(self, xs, ys, inds=None):
        # 获取 batch_size 和 seq_len
        b, t, d = xs.shape

        # 默认设置 inds
        if inds is None:
            inds = torch.arange(ys.shape[1], device=ys.device)
        else:
            inds = torch.as_tensor(inds, device=ys.device)

        # 将 xs 和 ys 交错并合并
        ys_wide = torch.cat([ys.view(b, t, 1), torch.zeros(b, t, d - 1, device=ys.device)], dim=2)
        zs = torch.stack([xs, ys_wide], dim=2).view(b, 2 * t, d)

        # 输入 LLaDA 模型
        input_ids = zs  # 使用 zs 作为 input_ids 传递给模型
        embeds = self._read_in(input_ids)  # 嵌入映射到模型的维度

        # 确保 embeds 维度正确 (b, seq_len, d_model)
        embeds = embeds.view(b, 2 * t, self.d_model)

        # 调用 _backbone 进行前向传播
        out = self._backbone(input_ids=input_ids, input_embeddings=embeds, output_hidden_states=True)  # 确保传递 input_ids
        last_h = out.hidden_states[-1]

        # 通过输出层获取预测
        pred = self._read_out(last_h)[:, ::2, 0][:, inds]

        return pred

# ========== 构建函数 ========== #
def build_model(conf):
    family = conf["family"]

    # 🧭 兼容不同命名 （s）
    n_layer = conf.get("n_layer", conf.get("n_layers", 6))
    n_head = conf.get("n_head", conf.get("n_heads", 8))
    n_embd = conf.get("n_embd", conf.get("d_model", 256))

    if family == "gpt2":
        return TransformerModel(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type="gpt2",
        # mlp_ratio=conf.get("mlp_ratio", 4.0), # 这个参数是否要统一

        )
    elif family == "gptJ":
        return TransformerModel(
            n_dims=conf["n_dims"],
            n_positions=conf["n_positions"],
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            type="gptJ",
            # mlp_ratio=conf.get("mlp_ratio", 4.0), # 这个参数是否要统一

        )
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
