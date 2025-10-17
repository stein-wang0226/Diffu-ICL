import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Any

from dllm.core.schedulers import BaseAlphaScheduler, LinearAlphaScheduler


class MDLMTrainer(transformers.Trainer):
    """
    Masked Diffusion Language Model Trainer.
    """

    def __init__(
        self,
        *args,
        scheduler: BaseAlphaScheduler | None = None,
        time_epsilon: float = 1e-3,
        loss_weight_type: str = "scheduler",  # "ones"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler or LinearAlphaScheduler()
        if not (0.0 < time_epsilon < 1.0):
            raise ValueError("time_epsilon must be in (0, 1)")
        self.time_epsilon = time_epsilon
        self.loss_weight_type = loss_weight_type

    def _preprocess_inputs(self, inputs):
        pass

    def _postprocess_outputs(self, outputs):
        pass

    def _compute_loss_weights(
        self,
        t: torch.Tensor,
        inputs: dict[str, Any],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute loss weights given timestep t and other arguments."""
        b, l = inputs["input_ids"].shape
        if self.loss_weight_type == "scheduler":
            loss_weights = -self.scheduler.weight(t).unsqueeze(1).repeat(1, l)  # b, 1
        elif self.loss_weight_type == "ones":
            loss_weights = torch.ones_like(inputs["input_ids"])
        else:
            raise NotImplementedError
        return loss_weights

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits = getattr(outputs, "logits", outputs)
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().contiguous()

        labels = inputs.get("labels")
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().contiguous()

        return (loss.detach(), logits, labels)

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        assert self.processing_class.padding_side == "right"
        self._preprocess_inputs(inputs)
        input_ids, labels, attention_mask = (
            inputs["input_ids"],
            inputs["labels"],
            inputs.get("attention_mask", None),
        )
        b, l = input_ids.shape

        # === 1. Sample diffusion timesteps ===
        # Each example draws a random timestep t ∈ [ε, 1), where ε avoids degenerate values near 0.
        # The scheduler defines the masking rate α(t); we convert it to a masking probability p_mask = 1 - α(t).
        t = self.time_epsilon + (1 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )
        p_mask = 1 - self.scheduler(t).unsqueeze(1).expand(b, l)

        # === 2. Apply stochastic masking ===
        # Tokens are masked independently according to p_mask(t).
        # Positions with label = -100 are excluded (ignored in loss).
        masked_indices = (torch.rand((b, l), device=input_ids.device) < p_mask) & (
            labels != -100
        )
        # Replace masked tokens with the special [MASK] token.
        noised_input_ids = torch.where(
            masked_indices, self.processing_class.mask_token_id, input_ids
        )

        # === 3. Forward pass through the model ===
        # The model predicts clean tokens given noised inputs.
        outputs = model(input_ids=noised_input_ids, attention_mask=attention_mask)
        self._postprocess_outputs(outputs)
        logits = outputs.logits

        # === 4. Handle degenerate cases (no tokens masked) ===
        # If no positions were masked, return a zero loss to keep gradients valid.
        # This step is necessary for Deepspeed Zero-{2,3}
        if not masked_indices.any():
            return (
                (logits.sum() * 0.0, outputs) if return_outputs else logits.sum() * 0.0
            )

        # === 5. Compute per-token loss weights ===
        # Depending on the configuration, weights may depend on timestep t
        # (e.g., scheduler-based) or be uniform (ones).
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_indices=masked_indices
        )

        # === 6. Compute weighted cross-entropy ===
        # Only masked tokens contribute to the loss.
        assert (input_ids[masked_indices] == labels[masked_indices]).all()
        token_loss = F.cross_entropy(
            logits[masked_indices], input_ids[masked_indices], reduction="none"
        )
        token_loss = token_loss * loss_weights[masked_indices]

        # === 7. Normalize loss per effective token length ===
        # Normalize each sequence’s contribution by its number of valid tokens,
        # then average over the batch for stability across variable-length inputs.
        effective_lengths = torch.sum(labels != -100, dim=1, keepdim=True).expand(b, l)
        loss = torch.sum(token_loss / effective_lengths[masked_indices]) / b

        # === 8. Return final loss (and optionally model outputs) ===
        return (loss, outputs) if return_outputs else loss
