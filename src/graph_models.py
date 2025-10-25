import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config, Qwen2Model, Qwen2Config
import math

from dllm.pipelines.dream.models.configuration_dream import DreamConfig
from dllm.pipelines.dream.models.modeling_dream import DreamBaseModel, DreamModel


def build_graph_model(conf):
    if conf.family == "gpt2":
        model = GraphTransformerModel(
            vocab_size=conf.vocab_size,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "qwen":
        model = GraphQwenModel(
            vocab_size=conf.vocab_size,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == "dream":
        model = GraphDreamModel(
            vocab_size=conf.vocab_size,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model

class GraphTransformerModel(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(GraphTransformerModel, self).__init__()
        configuration = GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"graph_gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self._read_in = nn.Embedding(configuration.vocab_size, n_embd, padding_idx=0)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    def forward(self, tokens, attention_mask=None, task_type="autoregressive"):
        embeds = self._read_in(tokens)
        output = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        
        # Aggregate token representations for prediction
        # We use masked averaging to ignore padding tokens
        if attention_mask is not None:
            masked_output = output * attention_mask.unsqueeze(-1)
            summed_output = masked_output.sum(dim=1)
            count = attention_mask.sum(dim=1, keepdim=True)
            # Avoid division by zero for empty sequences
            count = torch.max(count, torch.tensor(1.0, device=count.device))
            aggregated_output = summed_output / count
        else:
            aggregated_output = output.mean(dim=1)

        prediction = self._read_out(aggregated_output)
        return prediction.squeeze(-1)


class GraphQwenModel(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(GraphQwenModel, self).__init__()
        configuration = Qwen2Config(
            vocab_size=vocab_size,
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=False,
        )
        self.name = f"graph_qwen_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self._read_in = nn.Embedding(configuration.vocab_size, n_embd, padding_idx=0)
        self._backbone = Qwen2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    def forward(self, tokens, attention_mask=None, task_type="autoregressive"):
        embeds = self._read_in(tokens)
        output = self._backbone(inputs_embeds=embeds, attention_mask=attention_mask).last_hidden_state
        
        if attention_mask is not None:
            masked_output = output * attention_mask.unsqueeze(-1)
            summed_output = masked_output.sum(dim=1)
            count = attention_mask.sum(dim=1, keepdim=True)
            count = torch.max(count, torch.tensor(1.0, device=count.device))
            aggregated_output = summed_output / count
        else:
            aggregated_output = output.mean(dim=1)

        prediction = self._read_out(aggregated_output)
        return prediction.squeeze(-1)


class GraphDreamModel(nn.Module):
    def __init__(self, vocab_size, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(GraphDreamModel, self).__init__()
        self.family = "dream"
        configuration = DreamConfig(
            max_position_embeddings=n_positions,
            hidden_size=n_embd,
            intermediate_size=4 * n_embd,
            num_hidden_layers=n_layer,
            num_attention_heads=n_head,
            num_key_value_heads=n_head,
            use_cache=False,
        )
        self.name = f"graph_dream_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self._read_in = nn.Embedding(vocab_size, n_embd, padding_idx=0)
        self._backbone = DreamBaseModel(configuration)
        self._read_out = nn.Linear(n_embd, vocab_size)

    def forward(self, tokens, attention_mask=None, task_type="autoregressive"):
        extended_attention_mask = attention_mask
        if extended_attention_mask is not None:
            # The dream model's SDPA backend requires a float mask expanded to 4D.
            # Convert the incoming 2D boolean mask [batch, seq] to 4D float [batch, 1, 1, seq]
            extended_attention_mask = extended_attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min

        embeds = self._read_in(tokens)
        output = self._backbone(inputs_embeds=embeds, attention_mask=extended_attention_mask).last_hidden_state
        
        # Project to vocab size to get logits
        logits = self._read_out(output)
        
        return logits