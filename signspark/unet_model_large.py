import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Normalises over last dimension. x: (..., dim)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return (x / (rms + self.eps)) * self.scale


class RMSNorm1d(nn.Module):
    """RMSNorm for (B, C, T) tensors — normalises over channel dim."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(dim, eps)

    def forward(self, x):
        x = x.permute(0, 2, 1)    # B,C,T → B,T,C
        x = self.norm(x)
        return x.permute(0, 2, 1) # B,T,C → B,C,T


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # (T, dim//2) complex64


def apply_rope(q, k, freqs_cis):
    """q, k: (B, heads, T, head_dim)"""
    def rotate(x):
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        xc = torch.view_as_complex(x_)
        xr = xc * freqs_cis.unsqueeze(0).unsqueeze(0)
        return torch.view_as_real(xr).reshape(*x.shape).to(x.dtype)
    return rotate(q), rotate(k)


# ---------------------------------------------------------------------------
# AdaLN-Zero conditioning
# Predicts (shift, scale, gate) from the combined cond vector.
# Gate zero-init → each block starts as identity → stable flow matching training.
# ---------------------------------------------------------------------------

class AdaLNZero(nn.Module):
    """
    x: (B, C, T)
    cond: (B, cond_dim) — combined timestep+text embedding from time_mlp
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.proj = nn.Linear(cond_dim, 3 * dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)  # each (B, dim)
        x = x.permute(0, 2, 1)                                  # B,C,T → B,T,C
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = x * gate.unsqueeze(1).tanh()
        return x.permute(0, 2, 1)                               # B,T,C → B,C,T


# ---------------------------------------------------------------------------
# Conv building blocks
# ---------------------------------------------------------------------------

class Conv1dBlock(nn.Module):
    """Conv1d → RMSNorm → Mish. API-compatible with original."""
    def __init__(self, inp_channels, out_channels, kernel_size=5, n_groups=8, zero=False):
        super().__init__()
        self.conv = nn.Conv1d(inp_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.norm = RMSNorm1d(out_channels)
        self.act  = nn.Mish()
        if zero:
            nn.init.zeros_(self.conv.weight)
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Residual temporal block
# API-compatible with original ResidualTemporalBlock.
# adagn arg kept for compat — AdaLN-Zero is used regardless.
# ---------------------------------------------------------------------------

class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, embed_dim,
                 kernel_size=5, adagn=False, zero=False):
        super().__init__()
        self.conv1 = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, zero=zero)
        self.ada1  = AdaLNZero(out_channels, embed_dim)
        self.ada2  = AdaLNZero(out_channels, embed_dim)
        self.residual_proj = (
            nn.Conv1d(inp_channels, out_channels, 1)
            if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, x, cond):
        """x: (B, C, T), cond: (B, embed_dim)"""
        residual = self.residual_proj(x)
        h = self.conv1(x)
        h = h + self.ada1(h, cond)
        h = self.conv2(h)
        h = h + self.ada2(h, cond)
        return h + residual


# ---------------------------------------------------------------------------
# Temporal self-attention for down/up blocks
# ---------------------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """
    Self-attention over time with RoPE + RMSNorm.
    F.scaled_dot_product_attention dispatches to Flash Attention on PyTorch >= 2.0.
    x: (B, C, T)
    """
    def __init__(self, dim, heads=8, head_dim=64, dropout=0.0, max_seq_len=512):
        super().__init__()
        self.heads    = heads
        self.head_dim = head_dim
        inner_dim     = heads * head_dim

        self.norm   = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.drop   = nn.Dropout(dropout)

        self._rope_len = max_seq_len
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, max_seq_len),
            persistent=False,
        )

    def _get_freqs(self, T, device):
        if T > self._rope_len:
            self._rope_len = T
            self.freqs_cis = precompute_freqs_cis(self.head_dim, T, device=device)
        return self.freqs_cis[:T]

    def forward(self, x):
        B, C, T = x.shape
        residual = x

        x = x.permute(0, 2, 1)      # B,T,C
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.heads)

        freqs = self._get_freqs(T, x.device)
        q, k = apply_rope(q, k, freqs)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = rearrange(out, 'b h t d -> b t (h d)')
        out = self.drop(self.to_out(out)).permute(0, 2, 1)  # B,C,T

        return residual + out


# ---------------------------------------------------------------------------
# Transformer bottleneck block (self-attn + cross-attn to cond + SwiGLU FFN)
# Operates at fixed bottleneck_dim=1024.
# ---------------------------------------------------------------------------

class TransformerBottleneckBlock(nn.Module):
    """
    One transformer layer at the bottleneck.
    cond: (B, cond_dim) — treated as a single context token for cross-attention.
    x: (B, bottleneck_dim, T_bot)
    """
    def __init__(self, dim, cond_dim, heads=16, head_dim=64, ff_mult=4, dropout=0.0):
        super().__init__()
        inner_dim = heads * head_dim
        ff_dim    = int(dim * ff_mult)
        self.heads    = heads
        self.head_dim = head_dim

        # Self-attention
        self.norm_sa = RMSNorm(dim)
        self.to_qkv  = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_sa  = nn.Linear(inner_dim, dim, bias=False)

        # Cross-attention to conditioning vector
        self.norm_ca  = RMSNorm(dim)
        self.norm_ctx = RMSNorm(cond_dim)
        self.q_ca     = nn.Linear(dim, inner_dim, bias=False)
        self.kv_ca    = nn.Linear(cond_dim, inner_dim * 2, bias=False)
        self.out_ca   = nn.Linear(inner_dim, dim, bias=False)

        # SwiGLU FFN
        self.norm_ff = RMSNorm(dim)
        self.ff_gate = nn.Linear(dim, ff_dim, bias=False)
        self.ff_up   = nn.Linear(dim, ff_dim, bias=False)
        self.ff_down = nn.Linear(ff_dim, dim, bias=False)

        # AdaLN-Zero for each sub-block
        self.ada_sa = AdaLNZero(dim, cond_dim)
        self.ada_ca = AdaLNZero(dim, cond_dim)
        self.ada_ff = AdaLNZero(dim, cond_dim)

        self.drop = nn.Dropout(dropout)

        # RoPE for self-attention
        self._rope_len = 512
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, 512),
            persistent=False,
        )

    def _get_freqs(self, T, device):
        if T > self._rope_len:
            self._rope_len = T
            self.freqs_cis = precompute_freqs_cis(self.head_dim, T, device=device)
        return self.freqs_cis[:T]

    def forward(self, x, cond):
        """x: (B, C, T), cond: (B, cond_dim)"""
        B, C, T = x.shape
        x = x.permute(0, 2, 1)   # B,T,C
        ctx = cond.unsqueeze(1)   # B,1,cond_dim

        # --- Self-attention ---
        xn = self.norm_sa(x)
        q, k, v = self.to_qkv(xn).chunk(3, dim=-1)
        q = rearrange(q, 'b t (h d) -> b h t d', h=self.heads)
        k = rearrange(k, 'b t (h d) -> b h t d', h=self.heads)
        v = rearrange(v, 'b t (h d) -> b h t d', h=self.heads)
        freqs = self._get_freqs(T, x.device)
        q, k = apply_rope(q, k, freqs)
        sa = rearrange(F.scaled_dot_product_attention(q, k, v), 'b h t d -> b t (h d)')
        sa = self.drop(self.out_sa(sa))
        x = x + sa
        x = x.permute(0, 2, 1)
        x = x + self.ada_sa(x, cond)   # AdaLN gate in B,C,T space
        x = x.permute(0, 2, 1)         # back to B,T,C

        # --- Cross-attention to cond ---
        xn  = self.norm_ca(x)
        ctn = self.norm_ctx(ctx)
        q   = rearrange(self.q_ca(xn),  'b t (h d) -> b h t d', h=self.heads)
        k, v = self.kv_ca(ctn).chunk(2, dim=-1)
        k   = rearrange(k, 'b s (h d) -> b h s d', h=self.heads)
        v   = rearrange(v, 'b s (h d) -> b h s d', h=self.heads)
        ca  = rearrange(F.scaled_dot_product_attention(q, k, v), 'b h t d -> b t (h d)')
        ca  = self.drop(self.out_ca(ca))
        x = x + ca
        x = x.permute(0, 2, 1)
        x = x + self.ada_ca(x, cond)
        x = x.permute(0, 2, 1)

        # --- SwiGLU FFN ---
        xn = self.norm_ff(x)
        ff = self.ff_down(F.silu(self.ff_gate(xn)) * self.ff_up(xn))
        ff = self.drop(ff)
        x = x + ff
        x = x.permute(0, 2, 1)
        x = x + self.ada_ff(x, cond)

        return x   # B,C,T


# ---------------------------------------------------------------------------
# Projection into/out of bottleneck
# ---------------------------------------------------------------------------

class BottleneckProjection(nn.Module):
    """Projects encoder's final dim → fixed bottleneck_dim with normalisation."""
    def __init__(self, in_dim, bottleneck_dim=1024):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, bottleneck_dim, 1) if in_dim != bottleneck_dim else nn.Identity()
        self.norm = RMSNorm1d(bottleneck_dim)

    def forward(self, x):
        return self.norm(self.proj(x))


# ---------------------------------------------------------------------------
# Down / Up — same as original
# ---------------------------------------------------------------------------

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# TemporalUnetLarge
# ---------------------------------------------------------------------------

class TemporalUnetLarge(nn.Module):
    """
    Large modernised 1D UNet for flow matching sign language generation.

    Bottleneck is fixed at bottleneck_dim=1024 regardless of how wide the UNet
    gets, keeping it compact and reusable for downstream translation tasks.

    ---------------------------------------------------------
    bottleneck_dim        : int   — fixed bottleneck spatial dim (default 1024)
    num_bottleneck_layers : int   — transformer depth
    attn_resolutions      : tuple — depth indices that get temporal self-attn
    attn_heads            : int
    attn_head_dim         : int
    dropout               : float
    """

    def __init__(
        self,
        input_dim,
        cond_dim,
        dim=512,
        dim_mults=(1, 2, 4, 8),
        attention=True,
        adagn=True,
        zero=False,
        added_input_channels=0,
        out_mult=8,           # API compat
        final_type=2,         # API compat
        bottleneck_dim=1024,
        num_bottleneck_layers=8,
        attn_resolutions=(2, 3),
        attn_heads=16,
        attn_head_dim=64,
        dropout=0.0,
    ):
        super().__init__()

        dims = [input_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        mid_dim = dims[-1]

        print(f'[TemporalUnetLarge] channel dims : {dims}')
        print(f'[TemporalUnetLarge] bottleneck   : {bottleneck_dim}')
        print(f'[TemporalUnetLarge] attn at depth: {attn_resolutions}')

        time_dim = dim  # matches original convention

        # ── Conditioning MLP ─────────────────────────────────────────────
        # Same role as original time_mlp: (B, cond_dim) → (B, time_dim)
        # Deeper projection than original for more representational capacity.
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        # ── Encoder ──────────────────────────────────────────────────────
        self.downs = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last  = ind >= (num_resolutions - 1)
            is_first = ind == 0
            use_attn = attention and (ind in attn_resolutions)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(
                    dim_in + added_input_channels * is_first,
                    dim_out, embed_dim=time_dim, adagn=adagn, zero=zero,
                ),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, adagn=adagn, zero=zero),
                TemporalSelfAttention(
                    dim_out, heads=attn_heads, head_dim=attn_head_dim, dropout=dropout,
                ) if use_attn else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # ── Bottleneck ────────────────────────────────────────────────────
        self.proj_in  = BottleneckProjection(mid_dim, bottleneck_dim)
        self.proj_out = nn.Conv1d(bottleneck_dim, mid_dim, 1)

        self.mid_blocks = nn.ModuleList([
            TransformerBottleneckBlock(
                bottleneck_dim, cond_dim=time_dim,
                heads=attn_heads, head_dim=attn_head_dim,
                ff_mult=4, dropout=dropout,
            )
            for _ in range(num_bottleneck_layers)
        ])

        # ── Decoder ──────────────────────────────────────────────────────
        self.ups = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            depth_idx = num_resolutions - 2 - ind
            use_attn  = attention and (depth_idx in attn_resolutions)
            is_last   = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, adagn=adagn, zero=zero),
                TemporalSelfAttention(
                    dim_in, heads=attn_heads, head_dim=attn_head_dim, dropout=dropout,
                ) if use_attn else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        # ── Output ───────────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim_in, dim_in, kernel_size=5),
            nn.Conv1d(dim_in, input_dim, 1),
        )
        if zero:
            nn.init.zeros_(self.final_conv[1].weight)
            nn.init.zeros_(self.final_conv[1].bias)

        self.bottleneck = None  # populated during forward

        total = sum(p.numel() for p in self.parameters())
        print(f'[TemporalUnetLarge] Total parameters: {total / 1e9:.3f}B')

    def forward(self, x, cond):
        """
        x    : (seqlen, batch, input_dim)  — same layout as original TemporalUnet
        cond : (batch, cond_dim)           — combined timestep + text embedding
        Returns: (seqlen, batch, input_dim)
        """
        x = einops.rearrange(x, 's b d -> b d s')   # B,C,T

        # Project cond to time_dim — same role as original time_mlp
        c = self.time_mlp(cond)                      # B, time_dim

        # ── Encoder ──────────────────────────────────────────────────────
        skips = []
        for resnet1, resnet2, resnet3, attn, downsample in self.downs:
            x = resnet1(x, c)
            x = resnet2(x, c)
            x = resnet3(x, c)
            if not isinstance(attn, nn.Identity):
                x = attn(x)
            skips.append(x)
            x = downsample(x)

        # ── Bottleneck ───────────────────────────────────────────────────
        x = self.proj_in(x)                          # B, 1024, T_bot

        for i, block in enumerate(self.mid_blocks):
            x = block(x, c)
            if i == len(self.mid_blocks) // 2:
                self.bottleneck = x.clone()          # B, 1024, T_bot — for external use

        x = self.proj_out(x)                         # B, mid_dim, T_bot

        # ── Decoder ──────────────────────────────────────────────────────
        for resnet1, resnet2, resnet3, attn, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet1(x, c)
            x = resnet2(x, c)
            x = resnet3(x, c)
            if not isinstance(attn, nn.Identity):
                x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        return einops.rearrange(x, 'b d s -> s b d')  # S,B,D


# ---------------------------------------------------------------------------
# Kept from original for API compatibility
# ---------------------------------------------------------------------------

def cal_concat_multiple(in1, in2, multiple):
    a = (in1 + in2) / multiple
    return int((1 - (a - math.floor(a))) * multiple + in1 + in2)