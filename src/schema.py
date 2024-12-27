from quinine import (
    tstring,
    tinteger,
    tfloat,
    tboolean,
    stdict,
    tdict,
    default,
    required,
    allowed,
    nullable,
)
from funcy import merge
# 配置管理和参数校验 json

model_schema = {
    "family": merge(tstring, allowed(["gpt2", "lstm","gptJ"])),
    "n_positions": merge(tinteger, required),  # maximum context length
    "n_dims": merge(tinteger, required),  # latent dimension
    "n_embd": merge(tinteger, required),
    "n_layer": merge(tinteger, required),
    "n_head": merge(tinteger, required),
}

curriculum_base_schema = { #
    "start": merge(tinteger, required),  # initial parameter
    "end": merge(tinteger, required),  # limit of final value
    "inc": merge(tinteger, required),  # how much to increment each time
    "interval": merge(tinteger, required),  # increment every how many steps
}

curriculum_schema = {
    "dims": stdict(curriculum_base_schema),
    "points": stdict(curriculum_base_schema),
}

TASK_LIST = [ # add tasks
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
    # "linear_regression_uniform",
]

training_schema = {
    "task": merge(tstring, allowed(TASK_LIST)),
    "task_kwargs": merge(tdict, required),
    "num_tasks": merge(tinteger, nullable, default(None)),
    "num_training_examples": merge(tinteger, nullable, default(None)),
    "data": merge(tstring, allowed(["gaussian","uniform"])), # data distribution
    "w_type":merge(tstring,allowed(["gaussian","uniform"]),default("gaussian")), # task w distribution
    "batch_size": merge(tinteger, default(64)),
    "learning_rate": merge(tfloat, default(1e-4)),  #  large model 调大
    "weight_decay": merge(tfloat, default(0.00)), # for Adam
    "train_steps": merge(tinteger, default(1000)),
    "save_every_steps": merge(tinteger, default(1000)),  # how often to checkpoint
    "keep_every_steps": merge(tinteger, default(-1)),  # permanent checkpoints
    "resume_id": merge(tstring, nullable, default(None)),  # run uuid64
    "curriculum": stdict(curriculum_schema),
    "If_two_distribution": merge(tboolean, nullable, default(False)), # data 一半来自于distribution1 ， 一半 2
    "If_RandomShuffle_2distribution": merge(tboolean, nullable, default(False)),# 两类distribution 的batch打乱
    "w_distribution1": merge(tstring, allowed(["gaussian","uniform"]),default("gaussian")),
    "w_distribution2": merge(tstring, allowed(["gaussian", "uniform"]), default("uniform")),
}
eval_schema = {
    "If_shift_w_distribution": merge(tboolean, nullable, default(False)),
    "eval_w_type": merge(tstring, allowed(["add"]), default("add")),  # w1+w2
}
wandb_schema = {
    "project": merge(tstring, default("in-context-training")),
    "entity": merge(tstring, default("in-context")),
    "notes": merge(tstring, default("")),
    "name": merge(tstring, nullable, default(None)),
    "log_every_steps": merge(tinteger, default(10)),
}

schema = {
    "out_dir": merge(tstring, required),
    "model": stdict(model_schema),
    "training": stdict(training_schema),
    "eval": stdict(eval_schema),
    "wandb": stdict(wandb_schema),
    "test_run": merge(tboolean, default(False)), #
}
