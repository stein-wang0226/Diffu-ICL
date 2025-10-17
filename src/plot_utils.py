import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
# from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}



def basic_plot(metrics, models=None, trivial=1.0):
    # fig, ax = plt.subplots(1, 1, figsize=(20, 16))  # 宽 8 英寸，高 6 英寸
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if models is not None:
        metrics = {k: metrics[k] for k in models}
    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    # 设置标签和坐标轴范围
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("squared error")
    ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.1, 2.25)

    # 设置图例
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0)

    # 调整画布大小
    # fig.set_size_inches(20, 16)  # 增大画布大小
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    fig.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)  # 调整边距

    # 处理图例样式
    for line in legend.get_lines():
        line.set_linewidth(3)

    # 自动调整布局
    fig.tight_layout()

    return fig, ax

# def basic_plot(metrics, models=None, trivial=1.0):
#     # plt.figure(figsize=(8, 6))  # 设置画布大小
#     fig, ax = plt.subplots(1, 1)
#
#     if models is not None:
#         metrics = {k: metrics[k] for k in models}
#
#     color = 0
#     ax.axhline(trivial, ls="--", color="gray")
#     for name, vs in metrics.items():
#         ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
#         low = vs["bootstrap_low"]
#         high = vs["bootstrap_high"]
#         ax.fill_between(range(len(low)), low, high, alpha=0.3)
#         color += 1
#     ax.set_xlabel("in-context examples")
#     ax.set_ylabel("squared error")
#     ax.set_xlim(-1, len(low) + 0.1)
#     ax.set_ylim(-0.1, 1.25)
#
#     legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
#     fig.set_size_inches(8, 6)
#     for line in legend.get_lines():
#         line.set_linewidth(3)
#
#     return fig, ax


def collect_results(run_dir, df,w_type="add", valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path,w_type=w_type, skip_model_load=True) # todo

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    model_name = r.model
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
        # print(all_metrics)  # test
    return all_metrics
