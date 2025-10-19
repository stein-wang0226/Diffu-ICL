import os, sys, yaml, argparse, wandb, torch
from tqdm import tqdm
import numpy as np
import random
import yaml
from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler

# 确保能 import 到仓库根下的 dllm 包
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(repo_root)
sys.path.append(os.path.join(repo_root, "dllm"))

from models import build_model
from samplers import get_data_sampler
from tasks import get_task_sampler
from curriculum import Curriculum

# ---------------- utils ---------------- #
def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机种子 
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，也需要设置

set_seed(42) # 如果要可复现 

def load_yaml_config(path: str): 
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    out = model(xs, ys)   # 有时返回 pred，有时返回 (loss, pred)

    # ✅ 如果模型自己返回 loss（如 scheduler 模式）
    if isinstance(out, tuple) and len(out) == 2:
        loss, pred = out
    else:
        pred = out
        loss = loss_func(pred, ys)

    loss.backward()
    optimizer.step()
    return loss.detach().item(), pred.detach()


def sample_seeds(pool_size=None, bs=None, step=None): # 从种子池采样，增加随机性
    seeds = set()
    while len(seeds) < bs:
        seeds.add(random.randint(0, pool_size - 1))
    return seeds

# --------------- main train --------------- #

# def train(model, config):
#     print("=== Parsed config (brief) ===")
#     print("model:\n", yaml.dump(config["model"], sort_keys=False))
#     print("training:\n", yaml.dump(config["training"], sort_keys=False))
#     print("wandb:\n", yaml.dump(config["wandb"], sort_keys=False))
#     training = config["training"]
#     wandb_cfg = config["wandb"]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device).train()
#     model.hide_last_target = False  # ✅ 训练时允许看到全部 y
#     # model.predict_last_only=False 
#     # optimizer
#     optim = torch.optim.Adam(
#         model.parameters(),
#         lr=training["learning_rate"],
#         weight_decay=training["weight_decay"],
#     )
#     # curriculum
#     cur = Curriculum(training["curriculum"])
#     bsz = training["batch_size"]
#     n_dims = getattr(model, "n_dims", None) or model.n_dims
#     # data & task samplers
#     data_sampler = get_data_sampler(training["data"], n_dims=n_dims)
#     task_sampler = get_task_sampler(
#         training["task"],
#         n_dims,
#         bsz,
#         w_type=training["w_type"],
#         num_tasks=training.get("num_tasks"),  # 若为 1 则固定 w 的池
#         **training.get("task_kwargs", {}),
#     )

#     # I/O
#     # os.makedirs(config["out_dir"], exist_ok=True)
#     # state_path = os.path.join(config["out_dir"], "state.pt")
#     model_conf = config["model"]
#     train_conf = config["training"]
#     auto_dir = f"{model_conf['family']}_{train_conf['task']}_D{model_conf['n_dims']}_P{model_conf['n_positions']}_emb{model_conf['n_embd']}_bs{train_conf['batch_size']}"
#     out_dir = os.path.join("./checkpoints", auto_dir)
#     config["out_dir"] = out_dir  # ✅ 更新 config
#     os.makedirs(out_dir, exist_ok=True)
#     state_path = os.path.join(out_dir, "state.pt")
#     print(f"[Output Directory] {out_dir}")

#     # ===== resume logic =====
#     starting_step = 0
#     if os.path.exists(state_path):
#         print(f"[Resume] Found checkpoint at {state_path}, resuming training...")
#         state = torch.load(state_path, map_location=device)
#         model.load_state_dict(state["model_state_dict"])
#         optim.load_state_dict(state["optimizer_state_dict"])
#         starting_step = state.get("train_step", 0)
#         # 恢复 curriculum 的步数状态
#         for _ in range(starting_step + 1):
#             cur.update()
#         print(f"[Resume] Successfully resumed from step {starting_step}")

#     pool_size = training.get("num_training_examples", None)
#     pbar = tqdm(range(0, training["train_steps"]))
#     for step in pbar:
#         data_sampler_args = {}
#         task_sampler_args = {}
#         if pool_size is not None:  # 提供seed pool or not
#             assert pool_size >= bsz
#             seeds = sample_seeds(pool_size, bsz)
#             data_sampler_args["seeds"] = seeds
#             task_sampler_args["seeds"] = [s + 1 for s in seeds]

#         # === 采样 & 前向 === #
#         xs = data_sampler.sample_xs(cur.n_points, bsz, cur.n_dims_truncated, **data_sampler_args)  # [B, T, D]
#         task = task_sampler(**task_sampler_args)  # 注意：fixed w 情况下这里**不要**给 seeds
#         ys = task.evaluate(xs)  # [B, T]
#         # === 损失 === #
#         loss_func = task.get_training_metric()
#         # loss, output = train_step(model, xs, ys, optim, loss_func)
#         loss, output = train_step(model, xs.cuda(), 
#                 ys.cuda(), optim, 
#                 loss_func,
#                 # model.hide_last_target ,  # ✅ hide_last_target
#         )  # train update 参数 loss 为总误差/单点误差(ys)
#         # === 逐点 metric & baseline === #
#         pointwise_tags = list(range(cur.n_points))
#         pointwise_metric = task.get_metric()  # finer-grained
#         pointwise_loss = pointwise_metric(output, ys.cuda()).mean(dim=0)  # [T]
#         baseline_loss = (
#             sum(max(cur.n_dims_truncated - i, 0) for i in range(cur.n_points)) / cur.n_points
#         )
#         excess_loss = loss / baseline_loss if baseline_loss > 0 else loss
#         pointwise_loss_np = pointwise_loss.detach().cpu().flatten().numpy()

#         # === log === #
#         if step % wandb_cfg["log_every_steps"] == 0 :
#             wandb.log(
#                 {
#                     "overall_loss": loss,
#                     "excess_loss": excess_loss,
#                     "pointwise/loss": {str(i): float(v) for i, v in zip(pointwise_tags, pointwise_loss_np)},
#                     "curriculum/n_points": cur.n_points,
#                     "curriculum/n_dims": cur.n_dims_truncated,
#                 },
#                 step=step,
#             )

#         # === curriculum update === #
#         cur.update()
#         pbar.set_description(f"loss {loss:.6f} | dims={cur.n_dims_truncated} points={cur.n_points}")

#         # === save state === #
#         if step > 0 and step % training["save_every_steps"] == 0:
#             ckpt_state = {
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optim.state_dict(),
#                 "train_step": step,
#             }
#             torch.save(ckpt_state, state_path)

#         # === eval === #

#         if step % training["eval_every_steps"] == 0:
#             evaluation_kwargs = {
#                 "model": model,
#                 "task_name": training["task"],
#                 "data_name": training["data"],
#                 "n_dims": model.n_dims,
#                 "n_points": cur.n_points,
#                 "prompting_strategy": "standard",
#                 "num_eval_examples": 1280,
#                 "batch_size": 64,
#             }
#             from eval import eval_model
#             #########多点推理 version：原来eval时依次对每个y eval求mean as MSE （因为AR前向mask） ############
#             model.eval()
#             eval_metrics = eval_model(**evaluation_kwargs)
#             model.train()
#             eval_mean = np.mean(eval_metrics["mean"])
#             eval_std = np.mean(eval_metrics["std"])
#             eval_bootstrap_low = np.mean(eval_metrics["bootstrap_low"])
#             eval_bootstrap_high = np.mean(eval_metrics["bootstrap_high"])
#             wandb.log({
#                 "eval/mse": eval_mean,
#                 "eval/std_se": eval_std,
#                 "eval/bootstrap_low": eval_bootstrap_low,
#                 "eval/bootstrap_high": eval_bootstrap_high,
#             }, step=step)



# ---------------- train ---------------- #
def train(model, config):
    print("=== Parsed config (brief) ===")
    print("model:\n", yaml.dump(config["model"], sort_keys=False))
    print("training:\n", yaml.dump(config["training"], sort_keys=False))
    print("wandb:\n", yaml.dump(config["wandb"], sort_keys=False))

    training = config["training"]
    wandb_cfg = config["wandb"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    model.hide_last_target = False

    # ✅ 自动加载 LinearAlphaScheduler
    if hasattr(model, "mask_mode") and model.mask_mode == "scheduler":
        if not hasattr(model, "scheduler") or model.scheduler is None:
            model.scheduler = LinearAlphaScheduler()
            print("[Auto Scheduler] ✅ Using LinearAlphaScheduler for diffusion masking.")
    else:
        print("[Auto Scheduler] Skipped (mask_mode=fixed)")

    # optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    # curriculum
    cur = Curriculum(training["curriculum"])
    bsz = training["batch_size"]
    n_dims = getattr(model, "n_dims", None) or model.n_dims

    # data & task samplers
    data_sampler = get_data_sampler(training["data"], n_dims=n_dims)
    task_sampler = get_task_sampler(
        training["task"],
        n_dims,
        bsz,
        w_type=training["w_type"],
        num_tasks=training.get("num_tasks"),
        **training.get("task_kwargs", {}),
    )

    # ===== 自动输出目录 =====
    model_conf = config["model"]
    train_conf = config["training"]
    auto_dir = f"{model_conf['family']}_{train_conf['task']}_D{model_conf['n_dims']}_P{model_conf['n_positions']}_emb{model_conf['n_embd']}_bs{train_conf['batch_size']}"
    out_dir = os.path.join("./checkpoints", auto_dir)
    config["out_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    state_path = os.path.join(out_dir, "state.pt")
    print(f"[Output Directory] {out_dir}")

    # ===== Resume 逻辑 =====
    starting_step = 0
    if os.path.exists(state_path):
        print(f"[Resume] Found checkpoint at {state_path}, resuming training...")
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optim.load_state_dict(state["optimizer_state_dict"])
        starting_step = state.get("train_step", 0)
        for _ in range(starting_step + 1):
            cur.update()
        print(f"[Resume] Successfully resumed from step {starting_step}")

    # ===== 训练循环 =====
    pool_size = training.get("num_training_examples", None)
    pbar = tqdm(range(starting_step, training["train_steps"]))

    for step in pbar:
        data_sampler_args, task_sampler_args = {}, {}
        if pool_size is not None:
            assert pool_size >= bsz
            seeds = sample_seeds(pool_size, bsz)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # === 数据采样 ===
        xs = data_sampler.sample_xs(cur.n_points, bsz, cur.n_dims_truncated, **data_sampler_args)
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs)
        loss_func = task.get_training_metric()

        # === 前向 & 优化 ===
        loss, output = train_step(model, xs.to(device), ys.to(device), optim, loss_func)

        # === Logging ===
        pointwise_tags = list(range(cur.n_points))
        pointwise_metric = task.get_metric()
        pointwise_loss = pointwise_metric(output, ys.to(device)).mean(dim=0)
        baseline_loss = (
            sum(max(cur.n_dims_truncated - i, 0) for i in range(cur.n_points)) / cur.n_points
        )
        excess_loss = loss / baseline_loss if baseline_loss > 0 else loss
        pointwise_loss_np = pointwise_loss.detach().cpu().flatten().numpy()

        if step % wandb_cfg["log_every_steps"] == 0:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": excess_loss,
                    "pointwise/loss": {str(i): float(v) for i, v in zip(pointwise_tags, pointwise_loss_np)},
                    "curriculum/n_points": cur.n_points,
                    "curriculum/n_dims": cur.n_dims_truncated,
                },
                step=step,
            )

        cur.update()
        pbar.set_description(f"loss {loss:.6f} | dims={cur.n_dims_truncated} points={cur.n_points}")

        # === Save Checkpoint ===
        if step > 0 and step % training["save_every_steps"] == 0:
            ckpt_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "train_step": step,
            }
            torch.save(ckpt_state, state_path)

        # === Eval ===
        if step % training["eval_every_steps"] == 0:
            from eval import eval_model
            model.eval()
            evaluation_kwargs = {
                "model": model,
                "task_name": training["task"],
                "data_name": training["data"],
                "n_dims": model.n_dims,
                "n_points": cur.n_points,
                "prompting_strategy": "standard",
                "num_eval_examples": 1280,
                "batch_size": 64,
            }
            eval_metrics = eval_model(**evaluation_kwargs)
            model.train()

            eval_mean = np.mean(eval_metrics["mean"])
            eval_std = np.mean(eval_metrics["std"])
            eval_bootstrap_low = np.mean(eval_metrics["bootstrap_low"])
            eval_bootstrap_high = np.mean(eval_metrics["bootstrap_high"])
            wandb.log(
                {
                    "eval/mse": eval_mean,
                    "eval/std_se": eval_std,
                    "eval/bootstrap_low": eval_bootstrap_low,
                    "eval/bootstrap_high": eval_bootstrap_high,
                },
                step=step,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        notes=cfg["wandb"]["notes"],
        config=cfg,
        name=cfg["wandb"].get("name", None),
        resume="never",
    )

    model = build_model(cfg["model"])   # GPT 或 LLaDA 包装器
    train(model, cfg)

if __name__ == "__main__":
    main()
