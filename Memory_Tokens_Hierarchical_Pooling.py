import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence


class MemoryTokens(nn.Module):
    """ 
    Memory Tokens 记艺空间.

    Input:
        tokens: [B, N, D]
        padding_mask: [B, N], bool, True means padding token

    Output:
        reduced_tokens: [B, M, D]
    """

    def __init__(
        self,
        dim: int,
        num_memory_tokens: int = 8,
        depth: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_memory_tokens = num_memory_tokens

        self.memory_tokens = nn.Parameter(
            torch.empty(1, num_memory_tokens, dim)
        )
        nn.init.trunc_normal_(self.memory_tokens, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens should have shape [B, N, D], got {tokens.shape}")

        bsz = tokens.size(0)

        memory = self.memory_tokens.expand(bsz, -1, -1)
        x = torch.cat([tokens, memory], dim=1)

        src_key_padding_mask = None

        if padding_mask is not None:
            if padding_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    f"padding_mask should have shape [B, N], got {padding_mask.shape}"
                )

            memory_mask = torch.zeros(
                bsz,
                self.num_memory_tokens,
                dtype=torch.bool,
                device=tokens.device,
            )

            src_key_padding_mask = torch.cat(
                [padding_mask, memory_mask],
                dim=1,
            )

        out = self.encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        out = self.norm(out)

        reduced_tokens = out[:, -self.num_memory_tokens:, :]

        return reduced_tokens


class HierarchicalPooling(nn.Module):
    """
    Hierarchical Pooling module.

    Input:
        tokens: [B, N, D]
        padding_mask: [B, N], bool, True means padding token

    Output:
        reduced_tokens: [B, M, D]
    """

    def __init__(
        self,
        target_tokens: int = 8,
        protected_indices: Optional[Sequence[int]] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        if target_tokens <= 0:
            raise ValueError("target_tokens must be positive")

        self.target_tokens = target_tokens
        self.protected_indices = list(protected_indices or [])
        self.eps = eps

    def forward(
        self,
        tokens: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"tokens should have shape [B, N, D], got {tokens.shape}")

        if padding_mask is not None and padding_mask.shape != tokens.shape[:2]:
            raise ValueError(
                f"padding_mask should have shape [B, N], got {padding_mask.shape}"
            )

        outputs = []

        for b in range(tokens.size(0)):
            if padding_mask is None:
                valid_tokens = tokens[b]
                valid_positions = torch.arange(
                    tokens.size(1),
                    device=tokens.device,
                )
            else:
                keep = ~padding_mask[b]
                valid_tokens = tokens[b, keep]
                valid_positions = torch.arange(
                    tokens.size(1),
                    device=tokens.device,
                )[keep]

            out_b = self._pool_one_sample(
                valid_tokens=valid_tokens,
                valid_positions=valid_positions,
            )

            outputs.append(out_b)

        return torch.stack(outputs, dim=0)

    def _pool_one_sample(
        self,
        valid_tokens: torch.Tensor,
        valid_positions: torch.Tensor,
    ) -> torch.Tensor:
        dim = valid_tokens.size(-1)
        device = valid_tokens.device
        dtype = valid_tokens.dtype

        if valid_tokens.size(0) == 0:
            return valid_tokens.new_zeros(self.target_tokens, dim)

        protected_set = set(int(i) for i in self.protected_indices)

        protected_local = []
        poolable_local = []

        for local_idx, pos in enumerate(valid_positions.tolist()):
            if int(pos) in protected_set:
                protected_local.append(local_idx)
            else:
                poolable_local.append(local_idx)

        if len(protected_local) >= self.target_tokens:
            selected = protected_local[: self.target_tokens]
            return valid_tokens[selected]

        if len(protected_local) > 0:
            protected_tokens = valid_tokens[protected_local]
        else:
            protected_tokens = valid_tokens.new_empty(0, dim)

        if len(poolable_local) > 0:
            poolable_tokens = valid_tokens[poolable_local]
        else:
            poolable_tokens = valid_tokens.new_empty(0, dim)

        num_pool_tokens = self.target_tokens - protected_tokens.size(0)

        pooled_tokens = self._average_linkage_pool(
            tokens=poolable_tokens,
            target_count=num_pool_tokens,
        )

        reduced = torch.cat([pooled_tokens, protected_tokens], dim=0)

        return self._pad_or_trim(
            tokens=reduced,
            target_count=self.target_tokens,
            dim=dim,
            device=device,
            dtype=dtype,
        )

    def _average_linkage_pool(
        self,
        tokens: torch.Tensor,
        target_count: int,
    ) -> torch.Tensor:
        dim = tokens.size(-1)

        if target_count <= 0:
            return tokens.new_empty(0, dim)

        if tokens.size(0) == 0:
            return tokens.new_zeros(target_count, dim)

        if tokens.size(0) <= target_count:
            return self._pad_or_trim(
                tokens=tokens,
                target_count=target_count,
                dim=dim,
                device=tokens.device,
                dtype=tokens.dtype,
            )

        n = tokens.size(0)

        with torch.no_grad():
            z = F.normalize(
                tokens.detach(),
                p=2,
                dim=-1,
                eps=self.eps,
            )

            dist = 1.0 - z @ z.t()
            dist.fill_diagonal_(float("inf"))

            clusters = [[i] for i in range(n)]

            while len(clusters) > target_count:
                best_i = 0
                best_j = 1
                best_dist = float("inf")

                for i in range(len(clusters)):
                    idx_i = torch.tensor(
                        clusters[i],
                        device=tokens.device,
                    )

                    for j in range(i + 1, len(clusters)):
                        idx_j = torch.tensor(
                            clusters[j],
                            device=tokens.device,
                        )

                        d_ij = dist[
                            idx_i[:, None],
                            idx_j[None, :],
                        ].mean().item()

                        if d_ij < best_dist:
                            best_dist = d_ij
                            best_i = i
                            best_j = j

                clusters[best_i].extend(clusters[best_j])
                del clusters[best_j]

        pooled = []

        for cluster in clusters:
            idx = torch.tensor(
                cluster,
                device=tokens.device,
            )
            pooled.append(tokens[idx].mean(dim=0))

        return torch.stack(pooled, dim=0)

    @staticmethod
    def _pad_or_trim(
        tokens: torch.Tensor,
        target_count: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tokens.size(0) == target_count:
            return tokens

        if tokens.size(0) > target_count:
            return tokens[:target_count]

        pad = torch.zeros(
            target_count - tokens.size(0),
            dim,
            device=device,
            dtype=dtype,
        )

        return torch.cat([tokens, pad], dim=0)