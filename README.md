# 🧠 In-Context Learning + DLLM Integration Framework

This repository provides a unified framework that combines **In-Context Learning (ICL)** with **Diffusion Language Models (DLLM)**.  
It supports multiple model families — including **Transformer (GPT-2 / GPT-J)**, **LLaDA**, and **Dream** — enabling **fair and comparable experiments** between autoregressive and diffusion-based language models.

---

## 📂 Project Structure

**dllm需要import的主要是/llada/models/modelling_llada 下的模型class,**

**training的code 可以参考dllm.core.trainers下的逻辑自己写**

```
in-context-learning/
│
├── src/
│ ├── base_models.py # Base model definitions
│ ├── curriculum.py # Curriculum schedule for dynamic task scaling
│ ├── eval.py # Evaluation module
│ ├── models.py # Model building (GPT, LLaDA, Dream)
│ ├── samplers.py # Data sampler
│ ├── schema.py # Config schema
│ ├── tasks.py # Task definitions (e.g., regression, classification)
│ ├── train.py # Unified training loop
│ └── inference.py # Inference pipeline
│
└── dllm/ 
    └── dllm/
    ├── pipelines/
    │ ├── llada/
    │ │ ├── init.py
    │ │ ├── generate.py 
    │ │ └── trainer.py
    │ └── models/ modelling_llada.py # 这里是需要import LLaDa model 类的定义
    ├── examples/
    │ └── llada/
    │ ├── pt.py # Pretraining example
    │ └── sft.py # Finetuning example
    ├── utils/ # Helper functions for DLLM
    ├── data/ # Data processing
    └── core/ # 源码
          └── trainers/mdlm.py # diffusion train的主要逻辑

```

✅ **Supported Models**
- Transformer: GPT-2, GPT-J  ,Llama, ...
- LLaDA: Diffusion-based language model  
- Dream: Optional DLLM alternative

✅ **Unified API**
- Single `build_model` interface dynamically selects the model type.  
- Compatible with both AR and Diffusion training pipelines.

---

## 🧱 1. Model Definition (`src/models.py`)

We extend the baseline Transformer architecture to include DLLM models.

```python
from dllm.dllm.pipelines.llada.models.configuration_llada import LLaDAConfig
from dllm.dllm.pipelines.llada.models.modeling_llada import LLaDAModelLM as LLaDAModel

def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(...)
    elif conf.family == "gptJ":
        model = TransformerModel(...)
    elif conf.family == "llada":
        config = LLaDAConfig(
            d_model=conf.n_embd,
            n_heads=conf.n_head,
            n_layers=conf.n_layer,
            max_sequence_length=conf.n_positions,
            mlp_ratio=conf.mlp_ratio,
            use_cache=False,
        )
        model = LLaDAModel(config)
    elif conf.family == "dream":
        config = DreamConfig(...)
        model = DreamModel(config)
    else:
        raise NotImplementedError(f"Model family {conf.family} not supported.")
    return model
```

---

## 🏋️ 2. Training Pipeline (`src/train.py`)

This file unifies the **training loop** for both AR (GPT) and DLLM (LLaDA).  
You can **switch models** simply by changing the config file.

Key steps:
1. Load and build model from config  
2. Initialize data/task samplers and curriculum  
3. Run either `train_step_ar` or `train_step_llada`  
4. Log with Weights & Biases  
5. Save checkpoints regularly

```python
def train(model, config, is_llada=False):
    ...
    for step in tqdm(range(starting_step, train_steps)):
        xs = data_sampler.sample_xs(...)
        task = task_sampler()
        ys = task.evaluate(xs)

        if is_llada:
            input_ids = xs.long().to(device)
            loss, _ = train_step_llada(model, input_ids, optimizer)
        else:
            loss, _ = train_step_ar(model, xs.to(device), ys.to(device), optimizer, loss_func)
```

You can resume training from checkpoints using:

```bash
python src/train.py --config configs/llada.yaml
```

---

## ⚡ 3. Environment Setup

### 📌 Required Packages (DLLM)
```bash
transformers==4.57.0
accelerate==1.0.1
deepspeed==0.16.3
peft==0.13.2
bitsandbytes==0.42.0
datasets==3.2.0
sentencepiece==0.2.0
trl==0.23.1
tyro
wandb
omegaconf
tqdm
matplotlib
pytest
```

### 📌 ICL Environment
```yaml
name: in-context-learning
channels:
  - pytorch
  - defaults
dependencies:
  - pip=21.2.4
  - python=3.8.12
  - pytorch=1.11.0
  - pip:
    - jupyter==1.0.0
    - matplotlib==3.5.2
    - numpy==1.22.3
    - pandas==1.4.2
    - quinine          # install from source if needed
    - scikit-learn==1.0.2
    - seaborn==0.11.2
    - tqdm==4.64.0
    - transformers==4.17.0
    - wandb==0.12.11
    - xgboost==1.6.1
    - protobuf==3.20.1
```

---

## 🚀 4. Run Training

### ✅ Train with LLaDA (DLLM)
```bash
python src/train.py --config configs/llada.yaml
```

### ✅ Train with GPT-2
```bash
python src/train.py --config configs/gpt2.yaml
```

---

## 📊 5. Logging and Evaluation

- Training metrics are automatically logged with **Weights & Biases**.  
- Supports curriculum learning: `n_dims` and `n_points` can grow dynamically.  
- Evaluation scripts in `src/eval.py` can be extended for different tasks.

---

## 🧪 6. Key Features

- 🧠 Unified ICL Training Interface for AR and DLLM models  
- 🔀 Dynamic Curriculum Learning for scaling difficulty  
- 🧰 Extensible Task Samplers (regression, classification, custom)  
- 📝 Flexible Model Config through YAML  
- 🪄 WandB Integration for experiment tracking  
- 💾 Checkpointing for robust training resume

---

## 🧭 7. Roadmap

- [x] Integration of LLaDA model  
- [x] Curriculum training loop  
- [x] GPT / DLLM unified trainer  
- [ ] Tokenizer and prompt engineering for float input  
- [ ] Multi-task evaluation pipeline  
- [ ] Scaling to larger models (e.g., LLaDA-7B)

---

## 📝 Citation

If you find this repository useful, please consider citing:

```
@misc{dllm-icl,
  title  = {In-Context Learning + DLLM Integration Framework},
  author = {Yuxiang et al.},
  year   = {2025},
  note   = {https://github.com/...}
}
```

---

## 🤝 Acknowledgements

- LLaDA: Diffusion Language Model  
- PyTorch: Core deep learning framework  
- Weights & Biases: Logging and visualization  
- Inspired by works on in-context-learning and DLLM research.
