import torch

from torch import nn
from transformers import Trainer
from typing import List


class WeightedLossClassificationTrainer(Trainer):
    """Custom trainer for adding weighted cross entropy loss, useful for imbalanced dataset"""

    def set_loss_weights(self, weights: List[float]):
        """Set the weights for each class for the loss function
        Args:
            weights (List[float]): the weights for each class (if are present then weighted cross entropy loss is used)
        """
        self.loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(weights,
                                dtype=torch.float32,
                                device=self.model.device)
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """Define the metric for evaluation during training"""
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss
