# SPDX-License-Identifier: Apache-2.0
"""
Weight-map model architectures (master plan section 3).

Both PoC models predict a RESIDUAL IN DECISION SPACE on top of the B1 per-group
absmean baseline: the code head's output is added to a fixed-margin one-hot encoding
of the baseline code, and the scale head's output is added to log(baseline scale).
Final head layers are zero-initialized, so an untrained model reproduces the B1
baseline EXACTLY (a unit-tested property) - training can only be judged by what it
adds over the strong trivial rule, never silently lose to it at init.

    ContextMLP (A0, the null-hypothesis model): per-element MLP over the handcrafted
        context features - implemented as kernel-size-1 Conv1d for convenience, so it
        shares the [B, F, L] interface. No locality whatsoever: if nothing beats
        this, "QAT predictability is carried by simple statistics" is the finding.
    GroupConv1D (A1): 1-D convolutions ALONG THE COLUMN/GROUP AXIS within a row -
        the one axis where locality is defensible (128-column quantization groups;
        column stats are shared across rows) - with dilated residual blocks whose
        receptive field spans several groups. Per-row by construction; row
        adjacency is deliberately never used (rows are permutable, master plan 0.4).
"""
import torch
import torch.nn as nn

from ..data_utils.family_check import DEFAULT_GROUP_SIZE
from ..data_utils.features import NUM_FEATURES, feature_mask

BASELINE_LOGIT_MARGIN = 4.0
NUM_CODE_CLASSES = 3  # {-1, 0, +1} -> class index = code + 1
GROUP_CONV_RECEPTIVE_RADIUS = 88


def group_conv_context_halo(group_size: int) -> int:
    """Smallest group-aligned halo covering GroupConv1D's receptive radius."""
    return ((GROUP_CONV_RECEPTIVE_RADIUS + group_size - 1) // group_size) * group_size


class _FeatureMaskMixin:

    def _init_feature_mask(self, num_features: int, feature_set: str):
        if num_features != NUM_FEATURES and feature_set != 'full':
            raise ValueError('Named feature sets require the standard '
                             f'{NUM_FEATURES}-channel input.')
        mask = (feature_mask(feature_set) if num_features == NUM_FEATURES
                else torch.ones(num_features, dtype=torch.bool))
        # Derived configuration, deliberately absent from state_dict so historical
        # checkpoints load without missing-buffer errors.
        self.register_buffer('_input_feature_mask', mask.view(1, -1, 1),
                             persistent=False)
        self.feature_set = feature_set

    def _mask_features(self, features):
        if features.shape[1] != self._input_feature_mask.shape[1]:
            raise ValueError(f'Expected {self._input_feature_mask.shape[1]} feature '
                             f'channels, got {features.shape[1]}.')
        return features * self._input_feature_mask.to(features.dtype)


def baseline_code_logits(baseline_code: torch.Tensor,
                         margin: float = BASELINE_LOGIT_MARGIN) -> torch.Tensor:
    """[B, L] int code in {-1,0,1} -> [B, 3, L] fixed-margin one-hot logits."""
    one_hot = nn.functional.one_hot(baseline_code + 1, NUM_CODE_CLASSES)
    return margin * one_hot.permute(0, 2, 1).float()


class _Heads(nn.Module):
    """Shared zero-initialized code/scale residual heads over a [B, H, L] trunk."""

    def __init__(self, hidden: int, group_size: int):
        super(_Heads, self).__init__()
        self.group_size = group_size
        self.code_head = nn.Conv1d(hidden, NUM_CODE_CLASSES, kernel_size=1)
        self.scale_head = nn.Conv1d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.code_head.weight)
        nn.init.zeros_(self.code_head.bias)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def forward(self, trunk_out):
        code_delta = self.code_head(trunk_out)                     # [B, 3, L]
        bsz, hidden, seg_len = trunk_out.shape
        pooled = trunk_out.view(bsz, hidden, seg_len // self.group_size,
                                self.group_size).mean(dim=3)       # [B, H, L/G]
        scale_delta = self.scale_head(pooled).squeeze(1)           # [B, L/G]
        return code_delta, scale_delta


class ContextMLP(_FeatureMaskMixin, nn.Module):

    def __init__(self, num_features: int = NUM_FEATURES, hidden: int = 128,
                 num_layers: int = 3, group_size: int = DEFAULT_GROUP_SIZE,
                 feature_set: str = 'full'):
        super(ContextMLP, self).__init__()
        self._init_feature_mask(num_features, feature_set)
        layers = []
        in_ch = num_features
        for _ in range(num_layers):
            layers += [nn.Conv1d(in_ch, hidden, kernel_size=1), nn.GELU()]
            in_ch = hidden
        self.trunk = nn.Sequential(*layers)
        self.heads = _Heads(hidden, group_size)

    def forward(self, features):
        return self.heads(self.trunk(self._mask_features(features)))


class _DilatedResBlock(nn.Module):

    def __init__(self, hidden: int, kernel_size: int, dilation: int):
        super(_DilatedResBlock, self).__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding,
                               dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class GroupConv1D(_FeatureMaskMixin, nn.Module):

    def __init__(self, num_features: int = NUM_FEATURES, hidden: int = 128,
                 kernel_size: int = 9, dilations=(1, 4, 16),
                 group_size: int = DEFAULT_GROUP_SIZE,
                 feature_set: str = 'full'):
        super(GroupConv1D, self).__init__()
        self._init_feature_mask(num_features, feature_set)
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, hidden, kernel_size, padding=kernel_size // 2),
            nn.GELU())
        self.blocks = nn.Sequential(
            *[_DilatedResBlock(hidden, kernel_size, d) for d in dilations])
        self.heads = _Heads(hidden, group_size)

    def forward(self, features):
        features = self._mask_features(features)
        return self.heads(self.blocks(self.stem(features)))


def build_model(arch: str, num_features: int = NUM_FEATURES, hidden: int = 128,
                group_size: int = DEFAULT_GROUP_SIZE,
                feature_set: str = 'full') -> nn.Module:
    if arch == 'context_mlp':
        return ContextMLP(num_features, hidden, group_size=group_size,
                          feature_set=feature_set)
    if arch == 'group_conv':
        return GroupConv1D(num_features, hidden, group_size=group_size,
                           feature_set=feature_set)
    raise ValueError(f'Unknown arch {arch!r} (choices: context_mlp, group_conv).')


def predict_code_and_scales(model: nn.Module, batch: dict):
    """Compose model deltas with the batch's baseline: returns (code [B, L] in
    {-1,0,1}, scales [B, L/G], code_logits [B, 3, L], log_scales [B, L/G])."""
    code_delta, scale_delta = model(batch['features'])
    prediction_slice = batch.get('prediction_slice')
    if prediction_slice is not None:
        start, stop = prediction_slice
        code_delta = code_delta[:, :, start:stop]
        group_size = model.heads.group_size
        if start % group_size != 0 or stop % group_size != 0:
            raise ValueError('prediction_slice must be group-aligned.')
        scale_delta = scale_delta[:, start // group_size:stop // group_size]
    logits = baseline_code_logits(batch['baseline_code']) + code_delta
    log_scales = torch.log(batch['baseline_scales'].clamp(min=1e-12)) + scale_delta
    code = logits.argmax(dim=1) - 1
    return code, log_scales.exp(), logits, log_scales
