import os, sys, yaml, argparse, wandb, torch
from tqdm import tqdm
import numpy as np
import random

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


def train_step(model, xs, ys, optimizer, loss_func, predict_last_only=True):
    """
    单步训练逻辑：
    - 兼容 AR 模型（GPT系） 和 Diffusion 模型（LLaDA等）
    - 支持单点监督（predict_last_only=True）
    """
    optimizer.zero_grad()
    pred = model(xs, ys)  # 统一接口：GPT/LLaDA 都一样

    # ===== 公平性控制：只在AR模型上启用单点监督 =====
    if predict_last_only and hasattr(model, "family") and model.family in [
        "gpt2", "gptj", "qwen", "qwen2", "qwen2.5", "llama", "llama2", "llama3"
    ]:
        # 单点监督：仅监督最后一步预测
        loss = loss_func(pred[:, -1:], ys[:, -1:])
        # 可选打印一次确认
        # if random.random() < 0.0005:  # 每1000步左右打印一次
        #     print(f"[train_step] AR_last_only=True applied ({model.family}), using last-step supervision only.")
    else:
        # 默认全步监督
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

def train(model, config):
    print("=== Parsed config (brief) ===")
    print("model:", config["model"])
    print("training:", {k: v for k, v in config["training"].items() if k not in ["curriculum"]})
    print("wandb:", config["wandb"])

    training = config["training"]
    wandb_cfg = config["wandb"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).train()
    model.hide_last_target = False  # ✅ 训练时允许看到全部 y
    model.predict_last_only=True
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
        num_tasks=training.get("num_tasks"),  # 若为 1 则固定 w 的池
        **training.get("task_kwargs", {}),
    )

    # I/O
    os.makedirs(config["out_dir"], exist_ok=True)
    state_path = os.path.join(config["out_dir"], "state.pt")

    pool_size = training.get("num_training_examples", None)
    pbar = tqdm(range(0, training["train_steps"]))
    for step in pbar:
        data_sampler_args = {}
        task_sampler_args = {}
        if pool_size is not None:  # 提供seed pool or not
            assert pool_size >= bsz
            seeds = sample_seeds(pool_size, bsz)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # === 采样 & 前向 === #
        xs = data_sampler.sample_xs(cur.n_points, bsz, cur.n_dims_truncated, **data_sampler_args)  # [B, T, D]
        task = task_sampler(**task_sampler_args)  # 注意：fixed w 情况下这里**不要**给 seeds
        ys = task.evaluate(xs)  # [B, T]
        # xs, ys = xs.to(device), ys.to(device)
        # === 损失 === #
        loss_func = task.get_training_metric()
        # loss, output = train_step(model, xs, ys, optim, loss_func)
        loss, output = train_step(model, xs.cuda(), 
                ys.cuda(), optim, 
                loss_func,
                model.hide_last_target ,  # ✅ hide_last_target
        )  # train update 参数 loss 为总误差/单点误差(ys)
        # === 逐点 metric & baseline === #
        pointwise_tags = list(range(cur.n_points))
        pointwise_metric = task.get_metric()  # finer-grained
        pointwise_loss = pointwise_metric(output, ys.cuda()).mean(dim=0)  # [T]
        baseline_loss = (
            sum(max(cur.n_dims_truncated - i, 0) for i in range(cur.n_points)) / cur.n_points
        )
        excess_loss = loss / baseline_loss if baseline_loss > 0 else loss

        # === log === #
        if step % wandb_cfg["log_every_steps"] == 0 :
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": excess_loss,
                    "pointwise/loss": dict(
                        zip(pointwise_tags, pointwise_loss.cpu().numpy())
                    ),
                    "curriculum/n_points": cur.n_points,
                    "curriculum/n_dims": cur.n_dims_truncated,
                },
                step=step,
            )

        # === curriculum update === #
        cur.update()
        pbar.set_description(f"loss {loss:.6f} | dims={cur.n_dims_truncated} points={cur.n_points}")

        # === save state === #
        if step > 0 and step % training["save_every_steps"] == 0:
            ckpt_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "train_step": step,
            }
            torch.save(ckpt_state, state_path)

        # === eval === #

        if step % training["eval_every_steps"] == 0:
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
            from eval import eval_model
            # #########多点推理 version：原来eval时依次对每个y eval求mean as MSE （因为AR前向mask） ############
            # model.eval()
            # eval_metrics = eval_model(**evaluation_kwargs)
            # model.train()
            # eval_mean = np.mean(eval_metrics["mean"])
            # eval_std = np.mean(eval_metrics["std"])
            # eval_bootstrap_low = np.mean(eval_metrics["bootstrap_low"])
            # eval_bootstrap_high = np.mean(eval_metrics["bootstrap_high"])
            # wandb.log({
            #     "eval/mse": eval_mean,
            #     "eval/std_se": eval_std,
            #     "eval/bootstrap_low": eval_bootstrap_low,
            #     "eval/bootstrap_high": eval_bootstrap_high,
            # }, step=step)
            # ########多点推理 version #########

            
            ## ####单点推理version: 只预测最后一步 ######
            was_training = model.training
            model.eval()
            # ✅ 无论是 LLaDA 还是 AR 系列，都支持 hide_last_target/predict_last_only
            if hasattr(model, "hide_last_target"):
                old_hide = model.hide_last_target
                old_predict = model.predict_last_only

                # --- 在评估阶段隐藏最后一个 y_k，并只输出最后预测 ---
                model.hide_last_target = True
                model.predict_last_only = True
                print(f"[Eval] {model.name}: hide_last_target=True, predict_last_only=True")

            else:
                # 万一是未更新的旧模型（理论上不会出现）
                old_hide, old_predict = None, None
                print(f"[Eval] {model.name}: ⚠️ Model has no hide_last_target flag, eval may be unfair.")

            # 运行评估
            with torch.no_grad():
                eval_metrics = eval_model(**evaluation_kwargs)

            # 恢复状态
            if hasattr(model, "hide_last_target"):
                model.hide_last_target = old_hide
                model.predict_last_only = old_predict
            if was_training:
                model.train()

            # ✅ 只取最后一步yk的MSE指标
            eval_mean = float(np.mean(eval_metrics["mean"][-1]))
            eval_std = float(np.mean(eval_metrics["std"][-1]))
            eval_bootstrap_low = float(np.mean(eval_metrics["bootstrap_low"][-1]))
            eval_bootstrap_high = float(np.mean(eval_metrics["bootstrap_high"][-1]))

            wandb.log({
                "eval/mse_last": eval_mean,
                "eval/std_se": eval_std,
                "eval/bootstrap_low": eval_bootstrap_low,
                "eval/bootstrap_high": eval_bootstrap_high,
            }, step=step)

            ########### 单点推理version ###############

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
