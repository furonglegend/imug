# contrastive/prototype_contrastive.py
"""
Prototype-based contrastive loss with momentum-updated prototypes.

Usage:
    loss_module = PrototypeContrastiveLoss(num_prototypes, embed_dim, tau=0.07, momentum=0.99)
    loss = loss_module(embeddings, positive_proto_ids)  # embeddings: [B, D], positives: [B] integers
Notes:
    - Prototypes are stored as running vectors (torch.nn.Parameter is not required).
    - This module supports CPU/GPU and updates prototypes in-place.
"""
import torch
import torch.nn.functional as F
from torch import nn

class PrototypeContrastiveLoss(nn.Module):
    def __init__(self, num_prototypes, embed_dim, tau=0.07, momentum=0.99, device=None):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.embed_dim = int(embed_dim)
        self.tau = float(tau)
        self.momentum = float(momentum)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize prototypes randomly (L2-normalized)
        protos = torch.randn(self.num_prototypes, self.embed_dim, device=self.device)
        protos = F.normalize(protos, p=2, dim=1)
        # We keep prototypes as a buffer so they are moved with model.to(device)
        self.register_buffer("prototypes", protos)

    @torch.no_grad()
    def _momentum_update(self, proto_ids, embeddings):
        """
        Update prototypes[proto_ids] <- momentum * p + (1 - momentum) * emb
        Where embeddings are normalized.
        proto_ids: [B] ints
        embeddings: [B, D] normalized
        """
        p = self.prototypes[proto_ids]  # [B, D]
        updated = self.momentum * p + (1.0 - self.momentum) * embeddings
        updated = F.normalize(updated, p=2, dim=1)
        # scatter update (in-place)
        self.prototypes[proto_ids] = updated

    def forward(self, embeddings, positive_proto_ids):
        """
        embeddings: torch.Tensor [B, D]
        positive_proto_ids: torch.LongTensor [B] indices in [0, num_prototypes)
        Returns: scalar loss (torch.Tensor)
        """
        assert embeddings.dim() == 2 and embeddings.size(1) == self.embed_dim
        device = embeddings.device
        B = embeddings.size(0)

        # normalize embeddings
        emb_n = F.normalize(embeddings, p=2, dim=1)

        # gather prototypes
        protos = self.prototypes  # [P, D]

        # compute similarities: [B, P]
        sims = torch.matmul(emb_n, protos.t())  # cosine similarities in [-1,1]

        # scale by temperature
        logits = sims / self.tau

        # positives: index per sample -> logits[i, pos_id]
        pos_idx = positive_proto_ids.to(device).long()
        logits_pos = logits[torch.arange(B, device=device), pos_idx]  # [B]

        # compute contrastive NCE loss:
        # denom = sum_j exp(logits_ij); loss = -log( exp(logit_pos) / denom )
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(dim=1)
        numer = torch.exp(logits_pos)
        loss = -torch.log(numer / (denom + 1e-12))
        loss = loss.mean()

        # momentum update prototypes with current embeddings
        self._momentum_update(pos_idx, emb_n.detach())

        return loss
