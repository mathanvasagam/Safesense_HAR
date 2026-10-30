"""The paper-faithful Houssein-style tri-modal baseline.

The source paper discloses the operation types and the LSTM width, but not a
fully reproducible set of widths/dropout choices.  This implementation follows
the frozen project plan: each branch produces a 128-dimensional embedding,
which is fused by the paper's modality-level gate.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from .config import ExperimentConfig


class HousseinBaseline(nn.Module):
    """Minimal RGB-free depth/skeleton/IMU classifier.

    ``forward`` accepts cached tensors with shape ``[B, 32, channels]`` and
    optional frame-validity masks with shape ``[B, 32]``.  It returns
    ``(logits, modality_weights)`` with shapes ``[B, 27]`` and ``[B, 3]``.
    Branch dropout and classifier dropout default to the values reported in
    the paper's hyperparameter table.  The paper contains conflicting dropout
    descriptions, so these explicit defaults are a documented reproduction
    choice rather than an exact claim.
    """

    hidden_size = 128

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        *,
        branch_dropout: float = 0.12,
        classifier_dropout: float = 0.82,
        classifier_width: int = 128,
    ) -> None:
        super().__init__()
        config = config or ExperimentConfig()
        self.sequence_length = config.sequence_length
        self.depth_features = config.depth_features
        self.skeleton_features = config.skeleton_features
        self.imu_features = config.imu_features
        self.num_classes = config.num_classes
        if not 0 <= branch_dropout < 1 or not 0 <= classifier_dropout < 1:
            raise ValueError("dropout values must be in [0, 1)")
        if classifier_width < 1:
            raise ValueError("classifier_width must be positive")
        self.classifier_width = classifier_width

        self.depth_lstm = nn.LSTM(self.depth_features, self.hidden_size, batch_first=True)
        self.skeleton_conv = nn.Conv1d(self.skeleton_features, self.hidden_size, kernel_size=3, padding=1)
        self.skeleton_norm = nn.BatchNorm1d(self.hidden_size)
        self.skeleton_lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.imu_conv = nn.Conv1d(self.imu_features, self.hidden_size, kernel_size=3, padding=1)
        self.imu_norm = nn.BatchNorm1d(self.hidden_size)
        self.imu_lstm = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        self.branch_dropout = nn.Dropout(branch_dropout)
        # The paper specifies random-normal Conv1D initialization but omits
        # its scale; std=0.05 follows the common Keras default used here.
        for convolution in (self.skeleton_conv, self.imu_conv):
            nn.init.normal_(convolution.weight, mean=0.0, std=0.05)
            nn.init.zeros_(convolution.bias)

        # Eq. (6): Softmax(Linear2(Tanh(Linear1(F)))).
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size * 3),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 3, 3),
        )
        # Fig. 5 places a BatchNorm layer at the start of the Recognition block.
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 3),
            nn.Linear(self.hidden_size * 3, classifier_width),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_width, self.num_classes),
        )

    @staticmethod
    def _validate_stream(name: str, values: Tensor, features: int, sequence_length: int) -> None:
        if values.ndim != 3 or values.shape[1] != sequence_length or values.shape[2] != features:
            raise ValueError(
                f"{name} must have shape [B, {sequence_length}, {features}], got {tuple(values.shape)}"
            )

    @staticmethod
    def _validate_mask(name: str, mask: Tensor, batch: int, sequence_length: int) -> None:
        if mask.ndim != 2 or mask.shape != (batch, sequence_length):
            raise ValueError(f"{name} must have shape [B, {sequence_length}], got {tuple(mask.shape)}")

    @classmethod
    def _mask_or_ones(cls, name: str, mask: Optional[Tensor], values: Tensor) -> Tensor:
        batch, sequence_length = values.shape[:2]
        if mask is None:
            return values.new_ones((batch, sequence_length))
        cls._validate_mask(name, mask, batch, sequence_length)
        return mask.to(device=values.device, dtype=values.dtype).clamp(0, 1)

    @staticmethod
    def _masked_mean(sequence: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1)
        # The clamp keeps an entirely unavailable modality at a finite zero
        # embedding while preserving the regular mean for valid sequences.
        return (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _conv_branch(
        self,
        values: Tensor,
        mask: Tensor,
        conv: nn.Conv1d,
        norm: nn.BatchNorm1d,
        lstm: nn.LSTM,
    ) -> Tensor:
        masked_values = values * mask.unsqueeze(-1)
        features = conv(masked_values.transpose(1, 2))
        features = torch.relu(norm(features)).transpose(1, 2)
        features = features * mask.unsqueeze(-1)
        temporal, _ = lstm(features)
        return self._masked_mean(self.branch_dropout(temporal), mask)

    def forward(
        self,
        depth: Tensor,
        skeleton: Tensor,
        imu: Tensor,
        depth_mask: Optional[Tensor] = None,
        skeleton_mask: Optional[Tensor] = None,
        imu_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_stream("depth", depth, self.depth_features, self.sequence_length)
        self._validate_stream("skeleton", skeleton, self.skeleton_features, self.sequence_length)
        self._validate_stream("imu", imu, self.imu_features, self.sequence_length)
        if depth.shape[0] != skeleton.shape[0] or depth.shape[0] != imu.shape[0]:
            raise ValueError("all modality batches must have the same size")
        depth_mask = self._mask_or_ones("depth_mask", depth_mask, depth)
        skeleton_mask = self._mask_or_ones("skeleton_mask", skeleton_mask, skeleton)
        imu_mask = self._mask_or_ones("imu_mask", imu_mask, imu)

        depth_sequence, _ = self.depth_lstm(depth * depth_mask.unsqueeze(-1))
        depth_sequence = self.branch_dropout(depth_sequence)
        depth_embedding = self._masked_mean(depth_sequence, depth_mask)
        skeleton_embedding = self._conv_branch(
            skeleton, skeleton_mask, self.skeleton_conv, self.skeleton_norm, self.skeleton_lstm
        )
        imu_embedding = self._conv_branch(imu, imu_mask, self.imu_conv, self.imu_norm, self.imu_lstm)
        embeddings = torch.stack((depth_embedding, skeleton_embedding, imu_embedding), dim=1)
        concatenated = embeddings.flatten(start_dim=1)
        modality_weights = torch.softmax(self.gate(concatenated), dim=-1)
        weighted = (embeddings * modality_weights.unsqueeze(-1)).flatten(start_dim=1)
        return self.classifier(weighted), modality_weights
