# SPDX-License-Identifier: Apache-2.0
"""
Losses for the weight-map models (master plan section 3, corrections 1-2).

Code: class-weighted cross-entropy over {-1, 0, +1} (classification, NOT
MSE-on-fake-quantized weights - the published targets are an exact categorical code,
and CE keeps the eval metric and the loss aligned). The optional `hard_weight`
upweights entries where the B1 baseline already disagrees with the truth - the only
entries a residual-on-baseline model can actually improve.

Scale: Huber on log-scale, masked where the true group scale is 0 (all-zero groups
carry no scale information).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightMapLoss(nn.Module):

    def __init__(self, class_weights=(1.0, 1.0, 1.0), scale_weight: float = 0.1,
                 hard_weight: float = 1.0, huber_delta: float = 1.0):
        super(WeightMapLoss, self).__init__()
        self.register_buffer('class_weights', torch.tensor(class_weights))
        self.scale_weight = scale_weight
        self.hard_weight = hard_weight
        self.huber_delta = huber_delta

    def forward(self, code_logits, log_scales, batch):
        target = (batch['code'] + 1)                                   # [B, L] in {0,1,2}
        ce = F.cross_entropy(code_logits, target,
                             weight=self.class_weights.to(code_logits.dtype),
                             reduction='none')                         # [B, L]
        if self.hard_weight != 1.0:
            hard = (batch['baseline_code'] != batch['code']).float()
            ce = ce * (1.0 + (self.hard_weight - 1.0) * hard)
        code_loss = ce.mean()

        true_scales = batch['scales']
        mask = true_scales > 0
        if mask.any():
            scale_loss = F.huber_loss(log_scales[mask],
                                      torch.log(true_scales[mask]),
                                      delta=self.huber_delta)
        else:
            scale_loss = torch.zeros((), device=code_logits.device)

        total = code_loss + self.scale_weight * scale_loss
        return total, {'loss': float(total.item()), 'code_loss': float(code_loss.item()),
                       'scale_loss': float(scale_loss.item())}
