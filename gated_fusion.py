# fusion/gated_fusion.py
"""
Per-pair gating MLP for fusing multiple affinity channels or modality features.
This implements a small MLP that takes concatenated channel features and outputs soft weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerPairGating(nn.Module):
    def __init__(self, in_dim, n_channels=2, hidden_dim=64):
        """
        in_dim: dimensionality of per-channel features concatenated (or other meta features)
        n_channels: number of channels to weight (e.g., semantic, edit)
        """
        super().__init__()
        self.n_channels = n_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_channels)
        )

    def forward(self, meta_features):
        """
        meta_features: tensor [..., in_dim] (can be 2D batch or pairwise flattened)
        returns weights [..., n_channels] after softmax
        """
        logits = self.mlp(meta_features)
        weights = F.softmax(logits, dim=-1)
        return weights

class ChannelFusion:
    def __init__(self, gating_module):
        """
        gating_module: instance of PerPairGating
        """
        self.gating = gating_module

    def fuse(self, channel_arrays, meta_features=None):
        """
        channel_arrays: list of numpy arrays or tensors with same shape [...,]
                        e.g., two channels arrays of shape [N, M]
        meta_features: if provided, used to compute per-pair weights; otherwise equal weights applied.
        Returns fused array matching channel shapes.
        """
        import numpy as np
        # convert to torch tensors
        chs = [torch.tensor(c, dtype=torch.float32) for c in channel_arrays]
        shape = chs[0].shape
        stacked = torch.stack(chs, dim=-1)  # [..., n_channels]
        if meta_features is None:
            weights = torch.ones_like(stacked) / float(len(chs))
            fused = (stacked * weights).sum(dim=-1)
            return fused.numpy()
        else:
            # meta features expected to match flattened pair dimension; run gating
            # reshape stacked to [P, n_channels], meta_features [P, in_dim]
            flat_stacked = stacked.reshape(-1, stacked.shape[-1])
            if isinstance(meta_features, np.ndarray):
                meta_features = torch.tensor(meta_features, dtype=torch.float32)
            weights = self.gating(meta_features)  # [P, n_channels]
            fused_flat = (flat_stacked * weights).sum(dim=-1)
            fused = fused_flat.view(*shape).numpy()
            return fused
