"""PyTorch implementation of the xFactor architecture.

Faithful port of the original JAX implementation with fused self+global
MVLayer attention, latent pose representation, and pose-conditioned rendering.
"""

import copy
import os
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .loss import LossComputer

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None


# ──────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────


class _RMSNorm(nn.Module):
    """RMS normalisation with learnable per-head, per-dim scale."""

    def __init__(self, num_heads, head_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, x):
        # x: (..., num_heads, head_dim)
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (self.weight * x.float() / rms).to(x.dtype)


class _LayerScale(nn.Module):
    def __init__(self, dim, init_value=0.01):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x):
        return self.gamma * x


class _MLP2(nn.Module):
    """Two-layer FFN: Linear(4*features) -> GELU -> Linear(out_dim)."""

    def __init__(self, in_dim, features, out_dim, bias=True, small_init=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 4 * features, bias=bias)
        self.fc2 = nn.Linear(4 * features, out_dim, bias=bias)
        _init_linear(self.fc1, small_init, bias)
        _init_linear(self.fc2, small_init, bias)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _MLP3(nn.Module):
    """Three-layer FFN: Linear(4*f) -> GELU -> Linear(f) -> GELU -> Linear(out)."""

    def __init__(self, in_dim, features, out_dim, bias=True, small_init=False):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 4 * features, bias=bias)
        self.fc2 = nn.Linear(4 * features, features, bias=bias)
        self.fc3 = nn.Linear(features, out_dim, bias=bias)
        for fc in (self.fc1, self.fc2, self.fc3):
            _init_linear(fc, small_init, bias)

    def forward(self, x):
        return self.fc3(F.gelu(self.fc2(F.gelu(self.fc1(x)))))


def _init_linear(layer, small_init=False, has_bias=True):
    if small_init:
        nn.init.normal_(layer.weight, std=1e-6)
    else:
        nn.init.xavier_uniform_(layer.weight)
    if has_bias and layer.bias is not None:
        nn.init.normal_(layer.bias, std=1e-6)


# ──────────────────────────────────────────────────────────────────────
# Rotary position embedding (2-D sincos)
# ──────────────────────────────────────────────────────────────────────


def _get_2d_sincos_rope(head_dim, grid_h, grid_w, device, dtype):
    """Returns ``(H*W, head_dim, 2)`` with ``[cos, sin]`` in the last dim."""
    assert head_dim % 4 == 0
    omega = torch.arange(head_dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (head_dim / 4.0)))
    omega = omega.repeat_interleave(2)  # (head_dim // 2,)  each freq repeated for RoPE pairing

    gi = torch.arange(grid_h, device=device, dtype=torch.float32)
    gj = torch.arange(grid_w, device=device, dtype=torch.float32)
    gi, gj = torch.meshgrid(gi, gj, indexing="ij")
    gi, gj = gi.reshape(-1), gj.reshape(-1)

    angles = torch.cat([omega[None] * gi[:, None], omega[None] * gj[:, None]], -1)
    return torch.stack([angles.cos(), angles.sin()], -1).to(dtype)  # (N, D, 2)


def _rotate_half(x):
    """``[-x1, x0, -x3, x2, ...]`` interleave."""
    return torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)


def _apply_rope(x, sincos):
    """Apply 2-D RoPE.  x: (..., T, H, D),  sincos: (T, D, 2)."""
    cos = sincos[:, :, 0]  # (T, D)
    sin = sincos[:, :, 1]
    # Broadcast: (1,...,T,1,D)
    ndim_pre = x.ndim - 3  # number of leading batch dims before T
    for _ in range(ndim_pre):
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(-2)  # (..., T, 1, D)
    sin = sin.unsqueeze(-2)
    return x * cos + _rotate_half(x) * sin


# ──────────────────────────────────────────────────────────────────────
# Segment-based attention mask
# ──────────────────────────────────────────────────────────────────────


def _segment_mask(seg_ids):
    """seg_ids: (B, N) -> bool mask (B, 1, N, N)  True = attend."""
    # PyTorch `scaled_dot_product_attention` uses bool masks where `True`
    # means "masked / not allowed to attend".
    #
    # We want "only attend within the same segment id", so we mask
    # cross-segment pairs.
    return (seg_ids.unsqueeze(-1) != seg_ids.unsqueeze(-2)).unsqueeze(1)


# ──────────────────────────────────────────────────────────────────────
# MVLayer – fused per-view self + cross-view global attention
# ──────────────────────────────────────────────────────────────────────


class _MVLayer(nn.Module):
    def __init__(self, dim, num_heads, bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm1 = nn.LayerNorm(dim, bias=False)

        # 6 sets of QKV: global (q, k, v) + self (q, k, v)
        self.qkv = nn.Linear(dim, 6 * dim, bias=False)
        nn.init.xavier_uniform_(self.qkv.weight)

        # Per-head RMSNorm for Q, K
        self.q_norm = _RMSNorm(num_heads, self.head_dim)
        self.k_norm = _RMSNorm(num_heads, self.head_dim)
        self.qs_norm = _RMSNorm(num_heads, self.head_dim)
        self.ks_norm = _RMSNorm(num_heads, self.head_dim)

        # Fuse global + self -> dim
        self.out_proj = nn.Linear(2 * dim, dim, bias=bias)
        _init_linear(self.out_proj, small_init=False, has_bias=bias)
        self.ls1 = _LayerScale(dim)

        # FFN (note: original applies GELU *before* the MLP)
        self.norm2 = nn.LayerNorm(dim, bias=bias)
        self.ffn = _MLP2(dim, dim, dim, bias=bias)
        self.ls2 = _LayerScale(dim)

    def forward(self, x, seg_ids=None, sincos_rope=None):
        """
        x:           (B, V, T, C)
        seg_ids:     (B, T) or (B, V, T)   patch group IDs {0, 1}
        sincos_rope: (T, head_dim, 2)
        """
        B, V, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        if seg_ids is None:
            seg_ids = x.new_zeros(B, T, dtype=torch.long)
        if seg_ids.ndim == 2:
            seg_ids = seg_ids.unsqueeze(1).expand(-1, V, -1)  # (B, V, T)

        # ---- pre-norm + QKV ----
        x_n = self.norm1(x)
        qkv = self.qkv(x_n).reshape(B, V, T, 6, H, D)
        q, k, v, qs, ks, vs = qkv.unbind(3)  # each (B,V,T,H,D)

        q, k = self.q_norm(q), self.k_norm(k)
        qs, ks = self.qs_norm(qs), self.ks_norm(ks)

        if sincos_rope is not None:
            q = _apply_rope(q, sincos_rope)
            k = _apply_rope(k, sincos_rope)
            qs = _apply_rope(qs, sincos_rope)
            ks = _apply_rope(ks, sincos_rope)

        # ---- global (cross-view) attention ----
        q_g = q.reshape(B, V * T, H, D).transpose(1, 2)   # (B, H, VT, D)
        k_g = k.reshape(B, V * T, H, D).transpose(1, 2)
        v_g = v.reshape(B, V * T, H, D).transpose(1, 2)
        g_mask = _segment_mask(seg_ids.reshape(B, V * T))   # (B, 1, VT, VT)
        x_g = F.scaled_dot_product_attention(q_g, k_g, v_g, attn_mask=g_mask)
        x_g = x_g.transpose(1, 2).reshape(B, V, T, C)

        # ---- self (per-view) attention ----
        qs_f = qs.reshape(B * V, T, H, D).transpose(1, 2)  # (BV, H, T, D)
        ks_f = ks.reshape(B * V, T, H, D).transpose(1, 2)
        vs_f = vs.reshape(B * V, T, H, D).transpose(1, 2)
        s_mask = _segment_mask(seg_ids.reshape(B * V, T))    # (BV, 1, T, T)
        x_s = F.scaled_dot_product_attention(qs_f, ks_f, vs_f, attn_mask=s_mask)
        x_s = x_s.transpose(1, 2).reshape(B, V, T, C)

        # ---- merge + residual ----
        x0 = self.out_proj(torch.cat([x_g, x_s], dim=-1))
        x0 = self.ls1(x0) + x

        # ---- FFN (LN -> GELU -> MLP -> LayerScale -> residual) ----
        x1 = self.ffn(F.gelu(self.norm2(x0)))
        return self.ls2(x1) + x0


# ──────────────────────────────────────────────────────────────────────
# Pose head & encoder
# ──────────────────────────────────────────────────────────────────────


class _PoseHead(nn.Module):
    """Latent pose: ``proj(cat(z, z0)) - proj(cat(z0, z0))``."""

    def __init__(self, feat_dim, pose_dim):
        super().__init__()
        self.proj = _MLP3(feat_dim * 2, feat_dim, pose_dim)

    def forward(self, z, z0):
        """z, z0: (B, V, 2, C) -> (B, V, 2, pose_dim)"""
        p = torch.cat([z, z0], dim=-1)
        p0 = torch.cat([z0, z0], dim=-1)
        return self.proj(p) - self.proj(p0)


class _PoseEncoder(nn.Module):
    def __init__(self, dim, num_heads, num_layers, patch_size, pose_dim,
                 num_pose_tokens=32, use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.num_pose_tokens = num_pose_tokens

        self.patch_proj = nn.Linear(patch_size * patch_size * 3, dim)
        _init_linear(self.patch_proj, small_init=False, has_bias=True)

        self.pose_embed = nn.Parameter(torch.randn(num_pose_tokens, dim) * 0.02)
        self.pose_embed._no_weight_decay = True

        self.layers = nn.ModuleList(
            [_MVLayer(dim, num_heads) for _ in range(num_layers)]
        )
        self.use_checkpoint = use_checkpoint

        self.post_norm = nn.LayerNorm(dim)
        self.pose_head = _PoseHead(dim, pose_dim)

    def forward(self, x, pmask=None):
        """
        x:     (B, V, 3, H, W)  in [-1, 1]
        pmask: (B, N)  values {0, 1} or None
        Returns P  (B, V, 2, pose_dim) or (B, V, pose_dim) when pmask is None.
        """
        B, V, _, H, W = x.shape
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        N = Hp * Wp
        npt = self.num_pose_tokens

        # patchify + project
        tok = rearrange(x, "b v c (hp ph) (wp pw) -> b v (hp wp) (ph pw c)",
                        ph=ps, pw=ps)
        tok = self.patch_proj(tok)  # (B, V, N, C)

        return_single = pmask is None
        if return_single:
            pmask = tok.new_zeros(B, N, dtype=torch.long)
        pmask_v = pmask.unsqueeze(1).expand(-1, V, -1)  # (B, V, N)

        # append pose tokens (two copies for black / white groups)
        ptok = self.pose_embed.unsqueeze(0).unsqueeze(0).expand(B, V, -1, -1)
        z = torch.cat([tok, ptok, ptok], dim=2)  # (B, V, N+2*npt, C)

        seg = torch.cat([
            pmask_v,
            pmask_v.new_zeros(B, V, npt),
            pmask_v.new_ones(B, V, npt),
        ], dim=2)  # (B, V, N+2*npt)

        # RoPE: image patches get 2-D sincos; pose tokens get identity (cos=1, sin=0)
        rope = _get_2d_sincos_rope(self.head_dim, Hp, Wp, tok.device, tok.dtype)
        id_rope = torch.zeros(2 * npt, self.head_dim, 2,
                              device=tok.device, dtype=tok.dtype)
        id_rope[..., 0] = 1.0
        rope = torch.cat([rope, id_rope], dim=0)  # (N+2*npt, D, 2)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                z = torch_checkpoint(layer, z, seg, rope, use_reentrant=False)
            else:
                z = layer(z, seg, rope)

        # extract pose tokens -> (B, V, 2, npt, C) -> first token each
        z_p = z[:, :, N:].reshape(B, V, 2, npt, -1)[:, :, :, 0]  # (B, V, 2, C)
        z_p = F.gelu(self.post_norm(z_p))

        z0 = z_p[:, 0:1].expand_as(z_p)  # reference frame features
        P = self.pose_head(z_p, z0)       # (B, V, 2, pose_dim)

        if return_single:
            P = P[:, :, 0]  # (B, V, pose_dim)
        return P


# ──────────────────────────────────────────────────────────────────────
# Renderer (decoder)
# ──────────────────────────────────────────────────────────────────────


class _Renderer(nn.Module):
    def __init__(self, dim, num_heads, num_layers, patch_size, pose_dim,
                 use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size

        self.ctx_lift = _MLP2(patch_size * patch_size * 3 + pose_dim, dim, dim)
        self.tgt_lift = _MLP2(pose_dim, dim, dim)

        self.layers = nn.ModuleList(
            [_MVLayer(dim, num_heads) for _ in range(num_layers)]
        )
        self.use_checkpoint = use_checkpoint

        self.pixel_norm = nn.LayerNorm(dim)
        self.pixel_head = _MLP3(dim, dim, patch_size * patch_size * 3)

    def forward(self, x_ctx, P, pmask=None):
        """
        x_ctx: (B, V_ctx, 3, H, W)  in [-1, 1]
        P:     (B, V, 2, Cp) or (B, V, Cp)   V = V_ctx + 1 (last = target)
        pmask: (B, N) values {0, 1} or None
        Returns: (B, 3, H, W) in [-1, 1]
        """
        B, V_ctx, _, H, W = x_ctx.shape
        V = P.shape[1]  # V_ctx + 1
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        N = Hp * Wp

        patches = rearrange(x_ctx, "b v c (hp ph) (wp pw) -> b v (hp wp) (ph pw c)",
                            ph=ps, pw=ps)  # (B, V_ctx, N, ps²·3)

        if pmask is None:
            pmask = patches.new_zeros(B, N, dtype=torch.long)
        pmask_v = pmask.unsqueeze(1).expand(-1, V, -1)  # (B, V, N)

        # broadcast pose to patches via mask
        if P.ndim == 3:
            P0 = P1 = P.unsqueeze(2)  # (B, V, 1, Cp)
        else:
            P0, P1 = P[:, :, 0:1], P[:, :, 1:2]  # each (B, V, 1, Cp)
        mask_f = pmask_v.unsqueeze(-1).float()
        P_sp = (1.0 - mask_f) * P0 + mask_f * P1  # (B, V, N, Cp)

        # context tokens = patches + pose
        ctx = self.ctx_lift(torch.cat([patches, P_sp[:, :V_ctx]], dim=-1))
        tgt = self.tgt_lift(P_sp[:, -1])  # (B, N, C)
        z = torch.cat([ctx, tgt.unsqueeze(1)], dim=1)  # (B, V, N, C)

        rope = _get_2d_sincos_rope(self.head_dim, Hp, Wp, z.device, z.dtype)

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                z = torch_checkpoint(layer, z, pmask_v, rope, use_reentrant=False)
            else:
                z = layer(z, pmask_v, rope)

        z_tgt = z[:, -1]  # (B, N, C)
        rgb = self.pixel_head(F.gelu(self.pixel_norm(z_tgt)))  # (B, N, ps²·3)
        rgb = 2.0 * torch.sigmoid(rgb) - 1.0  # -> [-1, 1]
        return rearrange(rgb, "b (hp wp) (ph pw c) -> b c (hp ph) (wp pw)",
                         hp=Hp, wp=Wp, ph=ps, pw=ps, c=3)


# ──────────────────────────────────────────────────────────────────────
# Quadrant mask (with full_chance)
# ──────────────────────────────────────────────────────────────────────


def _quadrant_mask(B, Hp, Wp, device, full_chance=0.05):
    """Returns ``(pmask, qc)`` where pmask is ``(B, N)`` in {0,1}
    and qc is ``(B,)`` = sum of the 4 quadrant labels."""
    assert Hp % 2 == 0 and Wp % 2 == 0
    H2, W2 = Hp // 2, Wp // 2

    q = torch.zeros(B, 4, dtype=torch.long, device=device)
    q[:, :2] = 1  # two ones, two zeros
    full = torch.rand(B, device=device) <= full_chance
    q[full] = 0  # 5 % chance all zeros
    for i in range(B):
        q[i] = q[i, torch.randperm(4, device=device)]

    qc = q.sum(1)
    q00 = q[:, 0].view(B, 1, 1).expand(-1, H2, W2)
    q10 = q[:, 1].view(B, 1, 1).expand(-1, H2, W2)
    q01 = q[:, 2].view(B, 1, 1).expand(-1, H2, W2)
    q11 = q[:, 3].view(B, 1, 1).expand(-1, H2, W2)
    top = torch.cat([q00, q10], dim=2)
    bot = torch.cat([q01, q11], dim=2)
    return torch.cat([top, bot], dim=1).reshape(B, -1), qc


# ──────────────────────────────────────────────────────────────────────
# Data augmentation (colour jitter + Gaussian blur for pose encoder)
# ──────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _pose_augment(x, p_jitter=0.9, p_blur=0.3,
                  d_bright=0.2, d_cont=0.2, d_sat=0.2, d_hue=0.05,
                  blur_sigma_range=(0.1, 2.0), blur_ks=7):
    """Apply colour jitter + Gaussian blur to **all views** of each sample
    identically (time-invariant).

    x: (B, V, 3, H, W) in [-1, 1].  Returns same shape / range.
    """
    B, V, C, H, W = x.shape
    x01 = (x + 1.0) / 2.0  # -> [0, 1]
    x01 = x01.reshape(B, V * C, H, W)

    out = []
    for b in range(B):
        img = x01[b].reshape(V, C, H, W)  # (V, 3, H, W)
        if torch.rand(1).item() <= p_jitter:
            # brightness
            delta = (torch.rand(1).item() * 2 - 1) * d_bright
            img = (img + delta).clamp(0, 1)
            # contrast
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * d_cont
            mean = img.mean(dim=(-3, -2, -1), keepdim=True)
            img = ((img - mean) * factor + mean).clamp(0, 1)
            # saturation
            factor = 1.0 + (torch.rand(1).item() * 2 - 1) * d_sat
            gray = img.mean(dim=-3, keepdim=True)
            img = ((img - gray) * factor + gray).clamp(0, 1)
        if torch.rand(1).item() <= p_blur:
            sigma = blur_sigma_range[0] + torch.rand(1).item() * (
                blur_sigma_range[1] - blur_sigma_range[0]
            )
            ks = blur_ks
            ax = torch.arange(ks, device=x.device, dtype=x.dtype) - ks // 2
            kernel = torch.exp(-ax.pow(2) / (2 * sigma**2))
            kernel = kernel / kernel.sum()
            pad = ks // 2

            # Depthwise separable conv over (V*C) channels:
            # input:  [1, V*C, H, W]
            # weights: [V*C, 1, ks, 1] (vertical) / [V*C, 1, 1, ks] (horizontal)
            img_flat = img.reshape(1, V * C, H, W)

            # Horizontal pass (width)
            img_pad = F.pad(img_flat, (pad, pad, 0, 0), mode="reflect")
            kw = kernel.view(1, 1, 1, ks).repeat(V * C, 1, 1, 1)
            blur_out = F.conv2d(img_pad, kw, padding=0, groups=V * C)

            # Vertical pass (height)
            out_pad = F.pad(blur_out, (0, 0, pad, pad), mode="reflect")
            kh = kernel.view(1, 1, ks, 1).repeat(V * C, 1, 1, 1)
            blur_out = F.conv2d(out_pad, kh, padding=0, groups=V * C)

            img = blur_out.reshape(V, C, H, W).clamp(0, 1)
        out.append(img)
    out = torch.stack(out)  # (B, V, 3, H, W)
    return out * 2.0 - 1.0  # -> [-1, 1]


# ──────────────────────────────────────────────────────────────────────
# XFactor main model
# ──────────────────────────────────────────────────────────────────────


class XFactor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        m = config.model

        C = m.get("embed_dim", 1024)
        heads = m.get("num_heads", 16)
        pose_layers = m.get("aggregator_depth", 8)
        render_layers = m.get("decoder_depth", 8)
        ps = m.get("patch_size", 16)
        img_sz = m.get("image_size", 256)
        pose_dim = m.get("pose_dim", 256)
        # The original JAX implementation uses block_size // 2 = 32 global
        # tokens per partition for the stereo model.
        npt = m.get("num_pose_tokens", 32)

        self.patch_size = ps
        self.img_size = img_sz
        self.h_patches = img_sz // ps
        self.w_patches = img_sz // ps

        self.pose_enc = _PoseEncoder(
            C, heads, pose_layers, ps, pose_dim, npt,
        )
        self.render = _Renderer(
            C, heads, render_layers, ps, pose_dim,
        )

        # ---- LPIPS for training ----
        self._lpips_w = config.training.get("lpips_loss_weight", 0.5)
        if self._lpips_w > 0 and lpips_lib is not None:
            self._lpips = lpips_lib.LPIPS(net="vgg")
            self._lpips.eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)

        # ---- loss computer for inference ----
        self.loss_computer = LossComputer(config)

        # ---- training knobs ----
        self._self_decode_prob = m.get("self_decode_prob", 0.05)
        self._full_chance = m.get("full_chance", 0.05)
        self._pose_aug = config.training.get("pose_augmentation", True)

        self.config_bk = copy.deepcopy(config)

    # keep frozen modules in eval mode
    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()
        if hasattr(self, "_lpips"):
            self._lpips.eval()
        return self

    # ── forward dispatch ─────────────────────────────────────────────

    def forward(self, data, create_visual=False, render_video=False, iter=0):
        if self.training:
            return self._forward_train(data, create_visual=create_visual)
        return self._forward_inference(data, create_visual)

    # ══════════════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════════════

    def _forward_train(self, data, create_visual=False):
        imgs = data["image"]  # (B, V, 3, H, W) in [0, 1]
        B, V, _, H, W = imgs.shape
        device = imgs.device

        # normalise to [-1, 1]
        x = imgs * 2.0 - 1.0

        # ---- self-decode trick: small chance to decode reference ----
        smask = (torch.rand(B, device=device) <= self._self_decode_prob).float()
        x = x.clone()
        x[:, -1] = (1.0 - smask[:, None, None, None]) * x[:, -1] + \
                    smask[:, None, None, None] * x[:, 0]

        # ---- quadrant mask ----
        pmask, qc = _quadrant_mask(
            B, self.h_patches, self.w_patches, device, self._full_chance,
        )

        # ---- pose encoder (optionally augmented input) ----
        if self._pose_aug and self.training:
            x_enc = _pose_augment(x)
        else:
            x_enc = x
        P = self.pose_enc(x_enc, pmask)  # (B, V, 2, pose_dim)

        # ---- pose swap for transferability ----
        P0, P1 = P[:, :, 0], P[:, :, 1]  # each (B, V, Cp)
        q0 = (qc == 0).float()  # True when mask is all-zero (full_chance)
        q1 = (qc == 4).float()  # never True with 2+2 scheme

        P0n = (1.0 - q1[:, None, None]) * P0 + q1[:, None, None] * P1
        P1n = (1.0 - q0[:, None, None]) * P1 + q0[:, None, None] * P0
        P_swap = torch.stack([P1n, P0n], dim=2)  # (B, V, 2, Cp)

        # ---- render ----
        pred = self.render(x[:, :-1], P_swap, pmask)  # (B, 3, H, W) in [-1, 1]
        gt = x[:, -1]  # (B, 3, H, W)

        # ---- losses (in [-1, 1]) ----
        l1 = F.l1_loss(pred, gt)
        lpips_val = torch.tensor(0.0, device=device)
        if self._lpips_w > 0 and hasattr(self, "_lpips"):
            lpips_val = self._lpips(pred.float(), gt.float()).mean()
        loss = l1 + self._lpips_w * lpips_val

        # PSNR in [0, 1] space
        with torch.no_grad():
            mse01 = F.mse_loss((pred + 1) / 2, (gt + 1) / 2)
            psnr = -10.0 * torch.log10(mse01.clamp(min=1e-8))

        ret = edict(
            loss_metrics=edict(loss=loss, l1_loss=l1, lpips_loss=lpips_val, psnr=psnr),
        )

        if create_visual:
            Heff = self.h_patches * self.patch_size
            Weff = self.w_patches * self.patch_size
            pred01 = ((pred.detach() + 1) / 2).clamp(0, 1)
            ret.input = edict(
                image=imgs[:, 0:1, :, :Heff, :Weff].detach(),
                index=data.get("index", torch.zeros(B, V, 2, dtype=torch.long,
                                                    device=device))[:, 0:1],
                scene_name=data.get("scene_name", [""] * B),
            )
            ret.target = edict(
                image=imgs[:, -1:, :, :Heff, :Weff].detach(),
                index=data.get("index", torch.zeros(B, V, 2, dtype=torch.long,
                                                    device=device))[:, -1:],
                scene_name=data.get("scene_name", [""] * B),
            )
            ret.render = pred01.unsqueeze(1)  # (B, 1, 3, H, W)

        return ret

    # ══════════════════════════════════════════════════════════════════
    #  INFERENCE
    # ══════════════════════════════════════════════════════════════════

    def _forward_inference(self, data, create_visual=False):
        imgs = data["image"]  # (B, V, 3, H, W)
        B, V, _, H, W = imgs.shape
        device = imgs.device
        num_in = self.config.training.num_input_views

        x = imgs * 2.0 - 1.0  # -> [-1, 1]
        P = self.pose_enc(x, pmask=None)  # (B, V, pose_dim)

        Heff = self.h_patches * self.patch_size
        Weff = self.w_patches * self.patch_size

        rendered = []
        for tv in range(num_in, V):
            P_v = torch.cat([P[:, :num_in], P[:, tv : tv + 1]], dim=1)
            pred = self.render(x[:, :num_in, :, :Heff, :Weff], P_v, pmask=None)
            rendered.append(((pred + 1) / 2).clamp(0, 1))

        rendered = torch.stack(rendered, 1)  # (B, V_tgt, 3, H, W)
        tgt_gt = imgs[:, num_in:, :, :Heff, :Weff]

        loss_m = self.loss_computer(rendered, tgt_gt)

        inp_dict = edict(image=imgs[:, :num_in, :, :Heff, :Weff])
        tgt_dict = edict(image=tgt_gt)
        if "index" in data:
            inp_dict.index = data["index"][:, :num_in]
            tgt_dict.index = data["index"][:, num_in:]
        if "scene_name" in data:
            inp_dict.scene_name = data["scene_name"]
            tgt_dict.scene_name = data["scene_name"]

        return edict(
            loss_metrics=loss_m,
            render=rendered,
            input=inp_dict,
            target=tgt_dict,
        )

    # ── checkpoint loading ───────────────────────────────────────────

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            names = sorted(f for f in os.listdir(load_path) if f.endswith(".pt"))
            paths = [os.path.join(load_path, n) for n in names]
        else:
            paths = [load_path]
        try:
            ckpt = torch.load(paths[-1], map_location="cpu", weights_only=True)
        except Exception:
            traceback.print_exc()
            return None
        self.load_state_dict(ckpt["model"], strict=False)
        return 0

    def get_overview(self):
        c = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        return edict(
            pose_enc=c(self.pose_enc),
            render=c(self.render),
        )
