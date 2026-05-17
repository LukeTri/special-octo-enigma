#!/usr/bin/env python3
"""
Distance-banded attention experiment.

This script implements a modified attention score where the interaction between
query i and key j uses a distance-conditioned projection:

    score(i, j) = (P_{i-j} q_i)^T (P_{i-j} k_j)

Here, P_{i-j} is represented by band-specific dimensional truncation (prefix
projection) or by separate per-band Q/K projections. Bands are contiguous in
relative distance and causal.

The included benchmark is a synthetic associative recall task that stresses
long-range retrieval. It reports query-answer accuracy and distance-binned
accuracy.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import AuxRequest, create_block_mask, flex_attention

    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    AuxRequest = None
    create_block_mask = None
    flex_attention = None
    FLEX_ATTENTION_AVAILABLE = False


IGNORE_INDEX = -100
BatchData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclasses.dataclass(frozen=True)
class BandSpec:
    max_distance: Optional[int]  # None means +infinity
    dim: int


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_band_spec(spec: str) -> List[BandSpec]:
    """
    Parse a string like: "64:32,256:16,inf:8"
    meaning:
      distance 0..64   -> dim 32
      distance 65..256 -> dim 16
      distance 257..inf -> dim 8
    """
    out: List[BandSpec] = []
    last_max: Optional[int] = None
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Invalid band chunk '{chunk}'. Expected <max_distance>:<dim>.")
        max_dist_s, dim_s = chunk.split(":", 1)
        max_dist_s = max_dist_s.strip().lower()
        dim = int(dim_s.strip())
        if dim <= 0:
            raise ValueError(f"Band dim must be > 0, got {dim}.")

        if max_dist_s in ("inf", "infinity", "*"):
            max_dist = None
        else:
            max_dist = int(max_dist_s)
            if max_dist < 0:
                raise ValueError(f"max_distance must be >= 0 or inf, got {max_dist}.")

        if last_max is None:
            if out:
                raise ValueError("Only one 'inf' band can appear and it must be last.")
        else:
            if max_dist is None:
                pass
            elif max_dist <= last_max:
                raise ValueError(
                    f"Band max distances must be strictly increasing, got {max_dist} after {last_max}."
                )

        out.append(BandSpec(max_distance=max_dist, dim=dim))
        if max_dist is None:
            last_max = None
        else:
            last_max = max_dist

    if not out:
        raise ValueError("At least one band is required.")
    if any(b.max_distance is None for b in out[:-1]):
        raise ValueError("An 'inf' band can only appear in the final position.")
    return out


def format_band_spec(bands: Sequence[BandSpec]) -> str:
    chunks = []
    for b in bands:
        upper = "inf" if b.max_distance is None else str(b.max_distance)
        chunks.append(f"{upper}:{b.dim}")
    return ",".join(chunks)


def parse_value_band_spec(spec: str, attention_bands: Sequence[BandSpec]) -> Optional[List[BandSpec]]:
    if spec.strip().lower() in ("", "0", "off", "none", "false"):
        return None
    value_bands = parse_band_spec(spec)
    if len(value_bands) != len(attention_bands):
        raise ValueError(
            "--value-bands must have the same number of bands as --bands; "
            f"got {len(value_bands)} vs {len(attention_bands)}."
        )
    for idx, (score_band, value_band) in enumerate(zip(attention_bands, value_bands)):
        if score_band.max_distance != value_band.max_distance:
            raise ValueError(
                "--value-bands must use the same distance boundaries as --bands; "
                f"band {idx} has {value_band.max_distance} vs {score_band.max_distance}."
            )
    return value_bands


def band_coverage_masks(
    seq_len: int, bands: Sequence[BandSpec], device: torch.device
) -> List[torch.Tensor]:
    """
    Build causal band masks of shape [T, T], where mask[i, j] is True when
    key position j participates in the band for query position i.
    """
    pos = torch.arange(seq_len, device=device)
    dist = pos[:, None] - pos[None, :]
    causal = dist >= 0

    masks: List[torch.Tensor] = []
    lower = 0
    covered = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    max_d = seq_len - 1

    for b in bands:
        upper = max_d if b.max_distance is None else min(max_d, b.max_distance)
        if upper < lower:
            band_mask = torch.zeros_like(causal)
        else:
            band_mask = causal & (dist >= lower) & (dist <= upper)
        masks.append(band_mask)
        covered |= band_mask
        lower = upper + 1

        if b.max_distance is None:
            break

    if not torch.all(covered[causal]):
        missing = (~covered) & causal
        sample = torch.nonzero(missing)
        first = sample[0].tolist() if sample.numel() else ["?", "?"]
        raise ValueError(
            f"Bands do not cover all causal pairs. First missing [query,key]={first}."
        )
    return masks


class VanillaCausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self._mask_cache: Dict[Tuple[int, str], torch.Tensor] = {}

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, str(device))
        if key not in self._mask_cache:
            pos = torch.arange(seq_len, device=device)
            dist = pos[:, None] - pos[None, :]
            self._mask_cache[key] = dist >= 0
        return self._mask_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        causal = self._causal_mask(seqlen, x.device)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~causal[None, None, :, :], neg_inf)
        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        out = probs @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.resid_dropout(self.out_proj(out))


class DistanceBandedSelfAttention(nn.Module):
    """
    Distance-conditioned attention with contiguous causal bands.

    projection_mode:
      - "prefix": compute shared Q/K once and use first d_band channels.
      - "per_band": learn separate Q/K projections for each band.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bands: Sequence[BandSpec],
        projection_mode: str = "prefix",
        value_bands: Optional[Sequence[BandSpec]] = None,
        attention_backend: str = "dense",
        flex_block_size: int = 128,
        dropout: float = 0.0,
        learn_band_bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        if projection_mode not in ("prefix", "per_band"):
            raise ValueError("projection_mode must be one of: prefix, per_band")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.bands = list(bands)
        self.value_bands = list(value_bands) if value_bands is not None else None
        self.projection_mode = projection_mode
        if attention_backend not in ("dense", "flex"):
            raise ValueError("attention_backend must be one of: dense, flex")
        if attention_backend == "flex" and not FLEX_ATTENTION_AVAILABLE:
            raise RuntimeError("--attention-backend flex requested, but torch.nn.attention.flex_attention is unavailable")
        if attention_backend == "flex" and dropout != 0.0:
            raise ValueError("--attention-backend flex currently requires --dropout 0.0")
        if flex_block_size <= 0:
            raise ValueError(f"flex_block_size must be positive, got {flex_block_size}")
        self.attention_backend = attention_backend
        self.flex_block_size = flex_block_size

        for b in self.bands:
            if b.dim > self.head_dim:
                raise ValueError(
                    f"Band dim {b.dim} exceeds head_dim={self.head_dim}. "
                    "Dims are per head."
                )
        if self.value_bands is not None:
            for b in self.value_bands:
                if b.dim > self.head_dim:
                    raise ValueError(
                        f"Value band dim {b.dim} exceeds head_dim={self.head_dim}. "
                        "Dims are per head."
                    )

        if projection_mode == "prefix":
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        else:
            self.q_projs = nn.ModuleList(
                nn.Linear(d_model, n_heads * b.dim, bias=False) for b in self.bands
            )
            self.k_projs = nn.ModuleList(
                nn.Linear(d_model, n_heads * b.dim, bias=False) for b in self.bands
            )
            self.v_proj = None if self.value_bands is not None else nn.Linear(d_model, d_model, bias=False)

        if self.value_bands is not None:
            self.v_projs = nn.ModuleList(
                nn.Linear(d_model, n_heads * b.dim, bias=False) for b in self.value_bands
            )
            self.v_lifts = nn.ModuleList(
                nn.Linear(b.dim, self.head_dim, bias=False) for b in self.value_bands
            )
        else:
            self.v_projs = nn.ModuleList()
            self.v_lifts = nn.ModuleList()

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        if learn_band_bias:
            self.band_bias = nn.Parameter(torch.zeros(len(self.bands)))
        else:
            self.register_parameter("band_bias", None)

        self._mask_cache: Dict[Tuple[int, str], List[torch.Tensor]] = {}
        self._flex_mask_cache: Dict[Tuple[int, int, str, int], object] = {}

    def _band_masks(self, seq_len: int, device: torch.device) -> List[torch.Tensor]:
        key = (seq_len, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = band_coverage_masks(seq_len, self.bands, device)
        return self._mask_cache[key]

    def _flex_band_mask(self, seq_len: int, band_idx: int, device: torch.device):
        if create_block_mask is None:
            raise RuntimeError("FlexAttention block-mask creation is unavailable")
        key = (seq_len, band_idx, str(device), self.flex_block_size)
        if key not in self._flex_mask_cache:
            lower = 0 if band_idx == 0 else (self.bands[band_idx - 1].max_distance or 0) + 1
            upper = self.bands[band_idx].max_distance

            def mask_mod(_batch, _head, q_idx, kv_idx):
                dist = q_idx - kv_idx
                mask = dist >= lower
                if upper is not None:
                    mask = mask & (dist <= upper)
                return mask

            self._flex_mask_cache[key] = create_block_mask(
                mask_mod,
                None,
                None,
                seq_len,
                seq_len,
                device=device,
                BLOCK_SIZE=self.flex_block_size,
            )
        return self._flex_mask_cache[key]

    def _compute_prefix_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _flex_attention_with_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        value: torch.Tensor,
        band_idx: int,
        band_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if flex_attention is None or AuxRequest is None:
            raise RuntimeError("FlexAttention is unavailable")
        block_mask = self._flex_band_mask(q.size(2), band_idx, q.device)
        out, aux = flex_attention(
            q.contiguous(),
            k.contiguous(),
            value.contiguous(),
            block_mask=block_mask,
            scale=band_dim ** -0.5,
            return_aux=AuxRequest(lse=True),
        )
        if aux.lse is None:
            raise RuntimeError("FlexAttention did not return logsumexp statistics")
        return out, aux.lse

    def _forward_flex(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu" and torch.is_grad_enabled() and x.requires_grad:
            raise RuntimeError("--attention-backend flex does not support CPU backward; use --device cuda or --attention-backend dense")
        bsz, seqlen, _ = x.shape
        if self.projection_mode == "prefix":
            q_full, k_full, v_full = self._compute_prefix_qkv(x)
        else:
            q_full = None
            k_full = None
            if self.v_proj is None:
                v_full = None
            else:
                v_full = self.v_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        outputs: List[torch.Tensor] = []
        lses: List[torch.Tensor] = []
        for idx, band in enumerate(self.bands):
            if self.projection_mode == "prefix":
                if q_full is None or k_full is None:
                    raise RuntimeError("Prefix mode requires shared Q/K projections")
                q_band = q_full[..., : band.dim]
                k_band = k_full[..., : band.dim]
            else:
                q_band = (
                    self.q_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, band.dim)
                    .transpose(1, 2)
                )
                k_band = (
                    self.k_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, band.dim)
                    .transpose(1, 2)
                )

            if self.band_bias is not None:
                bias = self.band_bias[idx].to(dtype=q_band.dtype)

                def score_mod(score, _batch, _head, _q_idx, _kv_idx):
                    return score + bias

            else:
                score_mod = None

            if self.value_bands is None:
                if v_full is None:
                    raise RuntimeError("Full-dimensional value projection is not initialized")
                value = v_full
            else:
                value_band = self.value_bands[idx]
                value = (
                    self.v_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, value_band.dim)
                    .transpose(1, 2)
                )

            if flex_attention is None or AuxRequest is None:
                raise RuntimeError("FlexAttention is unavailable")
            block_mask = self._flex_band_mask(seqlen, idx, x.device)
            out, aux = flex_attention(
                q_band.contiguous(),
                k_band.contiguous(),
                value.contiguous(),
                score_mod=score_mod,
                block_mask=block_mask,
                scale=band.dim ** -0.5,
                return_aux=AuxRequest(lse=True),
            )
            if aux.lse is None:
                raise RuntimeError("FlexAttention did not return logsumexp statistics")
            if self.value_bands is not None:
                out = self.v_lifts[idx](out)
            outputs.append(out)
            lses.append(aux.lse)

        global_lse = torch.logsumexp(torch.stack(lses, dim=0), dim=0)
        out = torch.zeros(
            bsz,
            self.n_heads,
            seqlen,
            self.head_dim,
            device=x.device,
            dtype=outputs[0].dtype,
        )
        for band_out, lse in zip(outputs, lses):
            out = out + torch.exp(lse - global_lse).to(dtype=band_out.dtype)[..., None] * band_out
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.resid_dropout(self.out_proj(out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.attention_backend == "flex":
            return self._forward_flex(x)

        bsz, seqlen, _ = x.shape
        masks = self._band_masks(seqlen, x.device)

        if self.projection_mode == "prefix":
            q, k, v = self._compute_prefix_qkv(x)
        else:
            if self.v_proj is None:
                v = None
            else:
                v = self.v_proj(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        score_dtype = x.dtype
        scores = torch.full(
            (bsz, self.n_heads, seqlen, seqlen),
            fill_value=torch.finfo(score_dtype).min,
            device=x.device,
            dtype=score_dtype,
        )

        for idx, band in enumerate(self.bands):
            mask = masks[idx][None, None, :, :]
            d = band.dim
            if self.projection_mode == "prefix":
                q_band = q[..., :d]
                k_band = k[..., :d]
            else:
                q_band = (
                    self.q_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, d)
                    .transpose(1, 2)
                )
                k_band = (
                    self.k_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, d)
                    .transpose(1, 2)
                )
            band_scores = (q_band @ k_band.transpose(-2, -1)) * (d ** -0.5)
            if self.band_bias is not None:
                band_scores = band_scores + self.band_bias[idx]
            scores = torch.where(mask, band_scores, scores)

        probs = F.softmax(scores, dim=-1)
        probs = self.attn_dropout(probs)
        if self.value_bands is None:
            if v is None:
                raise RuntimeError("Full-dimensional value projection is not initialized")
            out = probs @ v
        else:
            out = torch.zeros(
                bsz,
                self.n_heads,
                seqlen,
                self.head_dim,
                device=x.device,
                dtype=x.dtype,
            )
            zero_prob = torch.zeros((), device=probs.device, dtype=probs.dtype)
            for idx, band in enumerate(self.value_bands):
                z = (
                    self.v_projs[idx](x)
                    .view(bsz, seqlen, self.n_heads, band.dim)
                    .transpose(1, 2)
                )
                band_probs = torch.where(masks[idx][None, None, :, :], probs, zero_prob)
                out = out + self.v_lifts[idx](band_probs.to(dtype=z.dtype) @ z).to(dtype=out.dtype)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_mult: int,
        dropout: float,
        attn_kind: str,
        bands: Sequence[BandSpec],
        value_bands: Optional[Sequence[BandSpec]],
        attention_backend: str,
        flex_block_size: int,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        if attn_kind == "baseline":
            self.attn = VanillaCausalSelfAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        elif attn_kind == "distance_prefix":
            self.attn = DistanceBandedSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                bands=bands,
                projection_mode="prefix",
                value_bands=value_bands,
                attention_backend=attention_backend,
                flex_block_size=flex_block_size,
                dropout=dropout,
            )
        elif attn_kind == "distance_per_band":
            self.attn = DistanceBandedSelfAttention(
                d_model=d_model,
                n_heads=n_heads,
                bands=bands,
                projection_mode="per_band",
                value_bands=value_bands,
                attention_backend=attention_backend,
                flex_block_size=flex_block_size,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attn_kind: {attn_kind}")

        hidden = mlp_mult * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        mlp_mult: int,
        dropout: float,
        attn_kind: str,
        bands: Sequence[BandSpec],
        value_bands: Optional[Sequence[BandSpec]],
        attention_backend: str,
        flex_block_size: int,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_mult=mlp_mult,
                dropout=dropout,
                attn_kind=attn_kind,
                bands=bands,
                value_bands=value_bands,
                attention_backend=attention_backend,
                flex_block_size=flex_block_size,
            )
            for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = idx.shape
        if seqlen > self.seq_len:
            raise ValueError(f"Input seqlen={seqlen} exceeds configured seq_len={self.seq_len}.")
        pos = torch.arange(seqlen, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@dataclasses.dataclass
class RecallTaskConfig:
    seq_len: int = 256
    num_pairs: int = 96
    key_vocab: int = 128
    value_vocab: int = 64
    far_bias: float = 0.7
    unique_keys: bool = True
    target_mode: str = "all_values"

    @property
    def vocab_size(self) -> int:
        return 3 + self.key_vocab + self.value_vocab

    @property
    def key_offset(self) -> int:
        return 3

    @property
    def value_offset(self) -> int:
        return 3 + self.key_vocab

    @property
    def bos_token(self) -> int:
        return 1

    @property
    def query_token(self) -> int:
        return 2

    @property
    def pad_token(self) -> int:
        return 0

    def required_length(self) -> int:
        # BOS + 2*num_pairs + QUERY + query_key + answer
        return 1 + 2 * self.num_pairs + 3


def sample_query_index(num_pairs: int, far_bias: float, rng: random.Random) -> int:
    """
    Pick the queried key index.
    Higher far_bias increases probability of querying earlier pairs, which
    increases query-to-key distance.
    """
    if num_pairs <= 1:
        return 0
    if rng.random() < far_bias:
        # Earlier key => farther distance to query position.
        return rng.randrange(0, max(1, num_pairs // 3))
    return rng.randrange(0, num_pairs)


def make_recall_batch(
    batch_size: int,
    cfg: RecallTaskConfig,
    device: torch.device,
    rng: random.Random,
) -> BatchData:
    if cfg.required_length() > cfg.seq_len:
        raise ValueError(
            f"seq_len={cfg.seq_len} is too short for num_pairs={cfg.num_pairs}. "
            f"Need at least {cfg.required_length()}."
        )
    if cfg.unique_keys and cfg.key_vocab < cfg.num_pairs:
        raise ValueError(
            f"unique_keys=True requires key_vocab >= num_pairs, got "
            f"key_vocab={cfg.key_vocab}, num_pairs={cfg.num_pairs}. "
            "Increase --key-vocab or pass --allow-key-repeats."
        )
    if cfg.target_mode not in ("final_only", "all_values"):
        raise ValueError("target_mode must be one of: final_only, all_values")

    x = torch.full((batch_size, cfg.seq_len), cfg.pad_token, dtype=torch.long, device=device)
    y = torch.full((batch_size, cfg.seq_len), IGNORE_INDEX, dtype=torch.long, device=device)
    answer_positions = torch.zeros((batch_size,), dtype=torch.long, device=device)
    query_distances = torch.zeros((batch_size,), dtype=torch.long, device=device)

    for b in range(batch_size):
        if cfg.unique_keys:
            keys = rng.sample(range(cfg.key_vocab), cfg.num_pairs)
        else:
            keys = [rng.randrange(0, cfg.key_vocab) for _ in range(cfg.num_pairs)]
        vals = [rng.randrange(0, cfg.value_vocab) for _ in range(cfg.num_pairs)]
        q_idx = sample_query_index(cfg.num_pairs, cfg.far_bias, rng=rng)

        seq: List[int] = [cfg.bos_token]
        key_positions: List[int] = []
        for i in range(cfg.num_pairs):
            key_positions.append(len(seq))
            seq.append(cfg.key_offset + keys[i])
            seq.append(cfg.value_offset + vals[i])

        query_pos = len(seq)
        seq.append(cfg.query_token)
        seq.append(cfg.key_offset + keys[q_idx])
        answer_token = cfg.value_offset + vals[q_idx]
        seq.append(answer_token)

        seq_t = torch.tensor(seq, dtype=torch.long, device=device)
        x[b, : seq_t.numel()] = seq_t

        if cfg.target_mode == "all_values":
            # Dense supervision teaches the per-sequence key->value bindings;
            # the final query still requires retrieval from earlier context.
            for i, key_pos in enumerate(key_positions):
                y[b, key_pos] = cfg.value_offset + vals[i]

        # Predict final answer token right after reading query key.
        answer_pred_pos = query_pos + 1
        y[b, answer_pred_pos] = answer_token
        answer_positions[b] = answer_pred_pos
        query_distances[b] = answer_pred_pos - key_positions[q_idx]

    return x, y, answer_positions, query_distances


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def distance_bin_name(low: int, high: Optional[int]) -> str:
    if high is None:
        return f"[{low},inf)"
    return f"[{low},{high})"


def build_eval_batches(
    cfg: RecallTaskConfig,
    device: torch.device,
    batch_size: int,
    num_batches: int,
    seed: int,
) -> List[BatchData]:
    rng = random.Random(seed)
    batches: List[BatchData] = []
    for _ in range(num_batches):
        batches.append(make_recall_batch(batch_size=batch_size, cfg=cfg, device=device, rng=rng))
    return batches


def evaluate_on_batches(
    model: TinyTransformerLM,
    cfg: RecallTaskConfig,
    batches: Sequence[BatchData],
    distance_bin_edges: Sequence[int],
    amp_dtype: Optional[torch.dtype] = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    total_correct = 0

    bins = list(distance_bin_edges)
    stats: Dict[str, List[int]] = {}
    for i in range(len(bins)):
        low = bins[i]
        high = bins[i + 1] if i + 1 < len(bins) else None
        name = distance_bin_name(low, high)
        stats[name] = [0, 0]  # correct, total

    with torch.no_grad():
        for x, y, answer_pos, dists in batches:
            if amp_dtype is not None and x.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(x)
            else:
                logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, cfg.vocab_size),
                y.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            total_loss += float(loss.item())

            batch_idx = torch.arange(x.size(0), device=x.device)
            pred = logits[batch_idx, answer_pos].argmax(dim=-1)
            tgt = y[batch_idx, answer_pos]
            correct = pred.eq(tgt)
            total_correct += int(correct.sum().item())
            total_items += int(tgt.numel())

            for i in range(len(bins)):
                low = bins[i]
                high = bins[i + 1] if i + 1 < len(bins) else None
                if high is None:
                    mask = dists >= low
                else:
                    mask = (dists >= low) & (dists < high)
                if mask.any():
                    name = distance_bin_name(low, high)
                    stats[name][0] += int(correct[mask].sum().item())
                    stats[name][1] += int(mask.sum().item())

    metrics: Dict[str, float] = {}
    metrics["loss"] = total_loss / max(1, total_items)
    metrics["answer_acc"] = total_correct / max(1, total_items)
    for name, (c, n) in stats.items():
        if n > 0:
            metrics[f"acc_{name}"] = c / n
    return metrics


def train_one(
    attn_kind: str,
    bands: Sequence[BandSpec],
    value_bands: Optional[Sequence[BandSpec]],
    args: argparse.Namespace,
    task_cfg: RecallTaskConfig,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    model_seed: int,
    train_data_seed: int,
    val_seed: int,
    test_seed: int,
) -> Dict[str, float]:
    set_global_seed(model_seed)
    effective_value_bands = value_bands if attn_kind != "baseline" else None
    effective_backend = args.attention_backend if attn_kind != "baseline" else "dense"
    model = TinyTransformerLM(
        vocab_size=task_cfg.vocab_size,
        seq_len=task_cfg.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_mult=args.mlp_mult,
        dropout=args.dropout,
        attn_kind=attn_kind,
        bands=bands,
        value_bands=effective_value_bands,
        attention_backend=effective_backend,
        flex_block_size=args.flex_block_size,
    ).to(device)

    if args.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    param_count = count_parameters(model)
    print(
        f"\n=== Training {attn_kind} ===\n"
        f"params={param_count:,} | bands={format_band_spec(bands)} | "
        f"value_bands={format_band_spec(effective_value_bands) if effective_value_bands is not None else 'full'} | "
        f"backend={effective_backend} | device={device.type} | amp={amp_dtype}"
    )

    scaler_enabled = device.type == "cuda" and amp_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    train_rng = random.Random(train_data_seed)
    val_batches = build_eval_batches(
        cfg=task_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        num_batches=args.eval_batches,
        seed=val_seed,
    )
    test_batches = build_eval_batches(
        cfg=task_cfg,
        device=device,
        batch_size=args.eval_batch_size,
        num_batches=args.eval_batches * 2,
        seed=test_seed,
    )

    eval_edges = [0, 32, 64, 128, 256, 512]
    for step in range(1, args.steps + 1):
        model.train()
        x, y, _, _ = make_recall_batch(
            batch_size=args.batch_size,
            cfg=task_cfg,
            device=device,
            rng=train_rng,
        )

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if (amp_dtype is not None and device.type == "cuda")
            else contextlib.nullcontext()
        )
        with amp_ctx:
            logits = model(x)
            loss = F.cross_entropy(
                logits.view(-1, task_cfg.vocab_size),
                y.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="mean",
            )

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            metrics = evaluate_on_batches(
                model=model,
                cfg=task_cfg,
                batches=val_batches,
                distance_bin_edges=eval_edges,
                amp_dtype=amp_dtype,
            )
            metric_bits = [f"step={step}", f"train_loss={loss.item():.4f}"]
            metric_bits.append(f"eval_loss={metrics['loss']:.4f}")
            metric_bits.append(f"eval_acc={metrics['answer_acc']:.4f}")
            for k in sorted(metrics.keys()):
                if k.startswith("acc_["):
                    metric_bits.append(f"{k}={metrics[k]:.4f}")
            print(" | ".join(metric_bits))

    final = evaluate_on_batches(
        model=model,
        cfg=task_cfg,
        batches=test_batches,
        distance_bin_edges=eval_edges,
        amp_dtype=amp_dtype,
    )
    final["params"] = float(param_count)
    return final


def print_summary(results: Dict[str, Dict[str, float]]) -> None:
    print("\n=== Final Summary ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        keys = ["params", "loss", "answer_acc"]
        keys.extend(sorted(k for k in metrics if k.startswith("acc_[")))
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if k == "params":
                    print(f"  {k}: {int(v):,}")
                else:
                    print(f"  {k}: {v:.6f}")

    if "baseline" in results and "distance_prefix" in results:
        delta = results["distance_prefix"]["answer_acc"] - results["baseline"]["answer_acc"]
        print(f"\nDelta (distance_prefix - baseline) answer_acc: {delta:+.6f}")
    if "baseline" in results and "distance_per_band" in results:
        delta = results["distance_per_band"]["answer_acc"] - results["baseline"]["answer_acc"]
        print(f"Delta (distance_per_band - baseline) answer_acc: {delta:+.6f}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distance-banded attention prototype experiment.")
    p.add_argument(
        "--mode",
        default="compare",
        choices=["compare", "baseline", "distance_prefix", "distance_per_band"],
        help="Which model to train. 'compare' runs baseline + both variants.",
    )
    p.add_argument(
        "--bands",
        default="64:16,256:8,inf:4",
        help="Band spec as 'max_dist:dim,max_dist:dim,...' where dim is per head.",
    )
    p.add_argument(
        "--value-bands",
        default="",
        dest="value_bands",
        help=(
            "Optional value bottleneck spec with the same distance boundaries as --bands, "
            "for example '64:16,256:8,inf:4'. Empty/off keeps full-dimensional values."
        ),
    )
    p.add_argument(
        "--attention-backend",
        default="dense",
        choices=["dense", "flex"],
        dest="attention_backend",
        help="Distance-attention backend. 'flex' uses FlexAttention block masks plus LSE recombination.",
    )
    p.add_argument(
        "--flex-block-size",
        type=int,
        default=128,
        dest="flex_block_size",
        help="Block size passed to FlexAttention block-mask construction.",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--data-seed",
        type=int,
        default=-1,
        help="Training-data RNG seed. Default -1 means reuse --seed.",
    )
    p.add_argument(
        "--val-seed",
        type=int,
        default=-1,
        help="Validation set seed. Default -1 means --seed + 1.",
    )
    p.add_argument(
        "--test-seed",
        type=int,
        default=-1,
        help="Test set seed. Default -1 means --seed + 2.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "bf16", "fp16"],
        help="Mixed precision policy. 'auto' enables bf16/fp16 on CUDA when possible.",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for potential GPU speedups (PyTorch 2.x).",
    )
    p.add_argument("--out-dir", type=str, default="runs", dest="out_dir")
    p.add_argument("--run-name", type=str, default="", dest="run_name")

    # Model
    p.add_argument("--d-model", type=int, default=128, dest="d_model")
    p.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    p.add_argument("--n-layers", type=int, default=4, dest="n_layers")
    p.add_argument("--mlp-mult", type=int, default=4, dest="mlp_mult")
    p.add_argument("--dropout", type=float, default=0.0)

    # Task
    p.add_argument("--seq-len", type=int, default=256, dest="seq_len")
    p.add_argument("--num-pairs", type=int, default=96, dest="num_pairs")
    p.add_argument("--key-vocab", type=int, default=128, dest="key_vocab")
    p.add_argument("--value-vocab", type=int, default=64, dest="value_vocab")
    p.add_argument(
        "--allow-key-repeats",
        action="store_true",
        help="Allow repeated keys inside one sequence. Default uses unique keys to avoid ambiguous retrieval targets.",
    )
    p.add_argument(
        "--target-mode",
        type=str,
        default="all_values",
        choices=["final_only", "all_values"],
        help="'all_values' trains on each key->value pair plus the final query; 'final_only' trains only final retrieval.",
    )
    p.add_argument(
        "--far-bias",
        type=float,
        default=0.7,
        help="Probability of selecting far-range query keys (0 to 1).",
    )

    # Optimization
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--eval-batch-size", type=int, default=128, dest="eval_batch_size")
    p.add_argument("--eval-batches", type=int, default=10, dest="eval_batches")
    p.add_argument("--log-interval", type=int, default=50, dest="log_interval")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    p.add_argument("--grad-clip", type=float, default=1.0, dest="grad_clip")
    return p


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_amp_dtype(precision: str, device: torch.device) -> Optional[torch.dtype]:
    if precision == "fp32":
        return None
    if device.type != "cuda":
        return None
    if precision == "bf16":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if precision == "fp16":
        return torch.float16
    # auto
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def resolve_seed(value: int, fallback: int) -> int:
    return fallback if value < 0 else value


def save_results(
    args: argparse.Namespace,
    task_cfg: RecallTaskConfig,
    bands: Sequence[BandSpec],
    value_bands: Optional[Sequence[BandSpec]],
    results: Dict[str, Dict[str, float]],
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name.strip() if args.run_name.strip() else f"{stamp}_{args.mode}_{device.type}"
    out_path = out_dir / f"{run_name}.json"
    payload = {
        "run_name": run_name,
        "timestamp": stamp,
        "device": device.type,
        "amp_dtype": str(amp_dtype),
        "bands": format_band_spec(bands),
        "value_bands": format_band_spec(value_bands) if value_bands is not None else None,
        "attention_backend": args.attention_backend,
        "flex_block_size": args.flex_block_size,
        "task": dataclasses.asdict(task_cfg),
        "args": vars(args),
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    args = build_argparser().parse_args()

    bands = parse_band_spec(args.bands)
    value_bands = parse_value_band_spec(args.value_bands, bands)
    device = choose_device(args.device)
    amp_dtype = choose_amp_dtype(args.precision, device)
    train_data_seed = resolve_seed(args.data_seed, args.seed)
    val_seed = resolve_seed(args.val_seed, args.seed + 1)
    test_seed = resolve_seed(args.test_seed, args.seed + 2)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    task_cfg = RecallTaskConfig(
        seq_len=args.seq_len,
        num_pairs=args.num_pairs,
        key_vocab=args.key_vocab,
        value_vocab=args.value_vocab,
        far_bias=args.far_bias,
        unique_keys=not args.allow_key_repeats,
        target_mode=args.target_mode,
    )
    if task_cfg.required_length() > task_cfg.seq_len:
        raise ValueError(
            f"Invalid task setup: seq_len={task_cfg.seq_len}, num_pairs={task_cfg.num_pairs}. "
            f"Need seq_len >= {task_cfg.required_length()}."
        )

    print("Distance-Banded Attention Experiment")
    print(f"bands={format_band_spec(bands)}")
    print(f"value_bands={format_band_spec(value_bands) if value_bands is not None else 'full'}")
    print(f"mode={args.mode}")
    print(
        f"backend={args.attention_backend} | flex_block_size={args.flex_block_size} | "
        f"device={device.type} | amp={amp_dtype} | torch_compile={args.torch_compile}"
    )
    print(
        f"seeds: model={args.seed} train_data={train_data_seed} "
        f"val={val_seed} test={test_seed}"
    )
    print(
        f"task: seq_len={task_cfg.seq_len}, num_pairs={task_cfg.num_pairs}, "
        f"key_vocab={task_cfg.key_vocab}, value_vocab={task_cfg.value_vocab}, "
        f"far_bias={task_cfg.far_bias}, unique_keys={task_cfg.unique_keys}, "
        f"target_mode={task_cfg.target_mode}"
    )

    run_order = (
        ["baseline", "distance_prefix", "distance_per_band"]
        if args.mode == "compare"
        else [args.mode]
    )
    results: Dict[str, Dict[str, float]] = {}
    for kind in run_order:
        results[kind] = train_one(
            attn_kind=kind,
            bands=bands,
            value_bands=value_bands,
            args=args,
            task_cfg=task_cfg,
            device=device,
            amp_dtype=amp_dtype,
            model_seed=args.seed,
            train_data_seed=train_data_seed,
            val_seed=val_seed,
            test_seed=test_seed,
        )

    print_summary(results)
    out_path = save_results(
        args=args,
        task_cfg=task_cfg,
        bands=bands,
        value_bands=value_bands,
        results=results,
        device=device,
        amp_dtype=amp_dtype,
    )
    print(f"\nSaved results JSON: {out_path}")


if __name__ == "__main__":
    main()
