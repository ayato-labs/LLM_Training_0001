from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from src.registry import MODEL_REGISTRY


@dataclass
class ModernGPTConfig:
    # Scale
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    block_size: int = 512
    ffn_mult: float = 8 / 3

    # Ablation flags
    use_qk_norm: bool = True
    use_value_residual: bool = True
    use_layernorm_scaling: bool = True
    use_per_head_gating: bool = True

    # RoPE
    rope_theta: float = 10000.0

    # Regularization
    z_loss_weight: float = 1e-4


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, self.inv_freq)
        return torch.cat([freqs, freqs], dim=-1)


def apply_rotary(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x * cos - rotate_half(x) * sin, x * sin + rotate_half(x) * cos], dim=-1)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.d_head = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.d_head, bias=False)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.d_head, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)
            self.qk_scale = nn.Parameter(torch.ones(1))

        if config.use_per_head_gating:
            self.gate_proj = nn.Linear(config.n_embd, config.n_head, bias=True)

        if config.use_value_residual:
            self.v_residual_alpha = nn.Parameter(torch.ones(1))
            self.v_residual_beta = nn.Parameter(torch.zeros(1))
            self.v_residual_scale = nn.Parameter(torch.ones(1))

        self.rotary = RotaryEmbedding(self.d_head, config.rope_theta)

    def forward(self, x, v_first: torch.Tensor | None = None):
        B, T, C = x.shape  # noqa: N806

        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)

        if self.config.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        cos_sin = self.rotary(T, x.device)
        cos = cos_sin.cos()
        sin = cos_sin.sin()
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        if self.config.use_value_residual and v_first is not None:
            a1, a2 = self.v_residual_alpha, self.v_residual_beta
            norm = (a1**2 + a2**2).sqrt().clamp(min=1e-6)
            v_first_kv = v_first.view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)
            v = self.v_residual_scale * (a1 * v + a2 * v_first_kv) / norm

        if self.n_kv_head != self.n_head:
            repeat = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        scale = self.qk_scale.item() if self.config.use_qk_norm else None
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )

        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        if self.config.use_per_head_gating:
            g = torch.sigmoid(self.gate_proj(x))
            attn = attn.view(B, T, self.n_head, self.d_head)
            attn = 2.0 * g.unsqueeze(-1) * attn
            attn = attn.view(B, T, C)

        return self.o_proj(attn)


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        hidden = int(config.n_embd * config.ffn_mult)
        hidden = (hidden + 63) // 64 * 64
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModernGPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        self.attn_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffn = SwiGLUFFN(config)

        self.ln_scale = 1.0 / ((layer_idx + 1) ** 0.5) if config.use_layernorm_scaling else 1.0

    def forward(self, x, v_first=None):
        normed = self.attn_norm(x) * self.ln_scale
        x = x + self.attn(normed, v_first=v_first)
        x = x + self.ffn(self.ffn_norm(x) * self.ln_scale)
        return x


@MODEL_REGISTRY.register("modern_gpt")
class ModernGPT(nn.Module):
    def __init__(self, config: ModernGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # noqa: N806
        x = self.tok_emb(idx)

        v_first = None
        for i, block in enumerate(self.blocks):
            if i == 0 and self.config.use_value_residual:
                v_first = block.attn.v_proj(block.attn_norm(x))
            x = block(x, v_first=v_first)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            if self.config.z_loss_weight > 0:
                z = torch.logsumexp(logits, dim=-1)
                loss = loss + self.config.z_loss_weight * z.pow(2).mean()

        return logits, loss
