import sys
import os
import copy
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange
from easydict import EasyDict as edict

from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.patch_embed import PatchEmbed
from vggt.layers.mlp import Mlp
from vggt.layers.layer_scale import LayerScale
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from vggt.heads.head_act import activate_pose
from vggt.utils.rotation import quat_to_mat

from .loss import LossComputer
from .xfactor import _MLP2, _MLP3, _MVLayer, _get_2d_sincos_rope

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Masked attention primitives
# ---------------------------------------------------------------------------

class MaskedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, proj_bias=True,
                 qk_norm=False, rope=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope

    def forward(self, x, pos=None, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MaskedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 proj_bias=True, ffn_bias=True, init_values=None,
                 qk_norm=False, rope=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskedAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            proj_bias=proj_bias, qk_norm=qk_norm, rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU, bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, pos=None, attn_mask=None):
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos, attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Pose predictor (iterative refinement, adapted from VGGT CameraHead)
# ---------------------------------------------------------------------------

class PosePredictor(nn.Module):
    """Canonical-plus-delta pose predictor (following RayZer / xFactor).

    First token is canonical (identity rotation, zero translation, predicted FoV).
    Remaining tokens receive a relative delta from a lightweight MLP.
    """

    def __init__(self, dim, target_dim=9, **kwargs):
        super().__init__()
        self.target_dim = target_dim
        self.token_norm = nn.LayerNorm(dim)
        self.rel_head = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, 7, bias=True),
        )
        self.fov_head = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, 1, bias=True),
        )
        self.fov_bias = 1.25
        for module in (self.rel_head, self.fov_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1e-3)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, pose_tokens, num_iterations=None):
        """
        Args:
            pose_tokens: [B, S, C] — first token is canonical (ctx).
        Returns:
            list of one pose encoding [B, S, 9] (absT_quaR_FoV)
        """
        B, S, C = pose_tokens.shape
        pose_tokens = self.token_norm(pose_tokens)
        device, dtype = pose_tokens.device, pose_tokens.dtype

        canonical = pose_tokens[:, 0:1]
        fov = self.fov_head(canonical[:, 0]).squeeze(-1) + self.fov_bias

        all_enc = torch.zeros(B, S, self.target_dim, device=device, dtype=dtype)
        all_enc[:, :, 3] = 1.0
        all_enc[:, :, 7] = fov[:, None]
        all_enc[:, :, 8] = fov[:, None]

        if S > 1:
            relative = pose_tokens[:, 1:]
            feat = torch.cat(
                [canonical.expand(-1, S - 1, -1), relative], dim=-1,
            )
            delta = self.rel_head(feat)
            all_enc[:, 1:, :7] = all_enc[:, 1:, :7] + delta

        result = activate_pose(
            all_enc, trans_act="linear", quat_act="linear", fl_act="relu",
        )
        return [result]


class LatentPoseHead(nn.Module):
    """xFactor-style latent pose head: proj(cat(z, z0)) - proj(cat(z0, z0))."""

    def __init__(self, feat_dim, pose_dim):
        super().__init__()
        self.proj = _MLP3(feat_dim * 2, feat_dim, pose_dim)

    def forward(self, z, z0):
        p = torch.cat([z, z0], dim=-1)
        p0 = torch.cat([z0, z0], dim=-1)
        return self.proj(p) - self.proj(p0)


class LatentRenderer(nn.Module):
    """xFactor-style renderer with optional encoder context features."""

    def __init__(self, dim, num_heads, num_layers, patch_size, pose_dim, use_checkpoint=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint

        self.ctx_feat_proj = nn.Linear(dim, dim, bias=True)
        self.ctx_lift = _MLP2(patch_size * patch_size * 3 + pose_dim, dim, dim)
        self.tgt_lift = _MLP2(pose_dim, dim, dim)
        self.layers = nn.ModuleList([_MVLayer(dim, num_heads) for _ in range(num_layers)])
        self.pixel_norm = nn.LayerNorm(dim)
        self.pixel_head = _MLP3(dim, dim, patch_size * patch_size * 3)

    def forward(self, ctx_feat, x_ctx, P, pmask=None):
        """
        ctx_feat: (B, V_ctx, N, C) encoder patch features or None
        x_ctx:    (B, V_ctx, 3, H, W) in [0, 1]
        P:        (B, V, 2, Cp) or (B, V, Cp), V = V_ctx + 1
        pmask:    (B, N) values {0, 1} or None
        Returns:  (B, 3, H, W) in [0, 1]
        """
        B, V_ctx, _, H, W = x_ctx.shape
        V = P.shape[1]
        ps = self.patch_size
        Hp, Wp = H // ps, W // ps
        N = Hp * Wp

        patches = rearrange(
            x_ctx, "b v c (hp ph) (wp pw) -> b v (hp wp) (ph pw c)", ph=ps, pw=ps
        )

        if pmask is None:
            pmask = patches.new_zeros(B, N, dtype=torch.long)
        pmask_v = pmask.unsqueeze(1).expand(-1, V, -1)

        if P.ndim == 3:
            P0 = P1 = P.unsqueeze(2)
        else:
            P0, P1 = P[:, :, 0:1], P[:, :, 1:2]
        mask_f = pmask_v.unsqueeze(-1).float()
        P_sp = (1.0 - mask_f) * P0 + mask_f * P1

        ctx = self.ctx_lift(torch.cat([patches, P_sp[:, :V_ctx]], dim=-1))
        if ctx_feat is not None:
            ctx = ctx + self.ctx_feat_proj(ctx_feat)
        tgt = self.tgt_lift(P_sp[:, -1])
        z = torch.cat([ctx, tgt.unsqueeze(1)], dim=1)

        rope = _get_2d_sincos_rope(self.head_dim, Hp, Wp, z.device, z.dtype)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                z = torch_checkpoint(layer, z, pmask_v, rope, use_reentrant=False)
            else:
                z = layer(z, pmask_v, rope)

        z_tgt = z[:, -1]
        rgb = self.pixel_head(F.gelu(self.pixel_norm(z_tgt)))
        rgb = torch.sigmoid(rgb)
        return rearrange(
            rgb, "b (hp wp) (ph pw c) -> b c (hp ph) (wp pw)",
            hp=Hp, wp=Wp, ph=ps, pw=ps, c=3,
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def generate_quadrant_mask(B, H_patches, W_patches, device,
                           full_chance=0.0, empty_chance=0.0):
    """Returns (pmask, qc) where pmask is [B, N] in {0,1}.

    By default samples follow the 2-visible / 2-masked quadrant split.
    ``full_chance`` forces all quadrants visible (qc == 0), while
    ``empty_chance`` forces all quadrants masked (qc == 4).
    """
    assert H_patches % 2 == 0 and W_patches % 2 == 0
    if full_chance < 0.0 or empty_chance < 0.0 or full_chance + empty_chance > 1.0:
        raise ValueError(
            f"Expected 0 <= full_chance + empty_chance <= 1, got "
            f"full_chance={full_chance}, empty_chance={empty_chance}"
        )
    H2, W2 = H_patches // 2, W_patches // 2
    q = torch.zeros(B, 4, dtype=torch.long, device=device)
    q[:, 2:] = 1
    mask_mode = torch.rand(B, device=device)
    full = mask_mode <= full_chance
    empty = (mask_mode > full_chance) & (mask_mode <= full_chance + empty_chance)
    q[full] = 0
    q[empty] = 1
    for i in range(B):
        q[i] = q[i, torch.randperm(4, device=device)]
    qc = q.sum(1)
    q00 = q[:, 0].view(B, 1, 1).expand(-1, H2, W2)
    q10 = q[:, 1].view(B, 1, 1).expand(-1, H2, W2)
    q01 = q[:, 2].view(B, 1, 1).expand(-1, H2, W2)
    q11 = q[:, 3].view(B, 1, 1).expand(-1, H2, W2)
    top = torch.cat([q00, q10], dim=2)
    bot = torch.cat([q01, q11], dim=2)
    pmask = torch.cat([top, bot], dim=1).reshape(B, -1)
    return pmask, qc


def cam_to_plucker(c2w, fxfycxcy, h, w):
    """c2w [B,4,4], fxfycxcy [B,4] (normalised) -> [B,6,H,W]."""
    B, device, dtype = c2w.shape[0], c2w.device, c2w.dtype
    ff = fxfycxcy.clone()
    ff[:, 0] *= w; ff[:, 1] *= h; ff[:, 2] *= w; ff[:, 3] *= h

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype), indexing="ij",
    )
    xx = xx[None].expand(B, -1, -1).reshape(B, -1)
    yy = yy[None].expand(B, -1, -1).reshape(B, -1)
    xx = (xx + 0.5 - ff[:, 2:3]) / ff[:, 0:1]
    yy = (yy + 0.5 - ff[:, 3:4]) / ff[:, 1:2]
    zz = torch.ones_like(xx)
    d = torch.stack([xx, yy, zz], 2)
    d = torch.bmm(d, c2w[:, :3, :3].transpose(1, 2))
    d = d / d.norm(dim=2, keepdim=True)
    o = c2w[:, :3, 3][:, None, :].expand_as(d)
    o = o.reshape(B, h, w, 3).permute(0, 3, 1, 2)
    d = d.reshape(B, h, w, 3).permute(0, 3, 1, 2)
    return torch.cat([torch.cross(o, d, dim=1), d], dim=1)


def pose_enc_to_c2w_fxfycxcy(pose_enc, image_hw):
    """pose_enc [B, S, 9] -> c2w [B*S,4,4], fxfycxcy [B*S,4] (normalised).

    Avoids torch.linalg.inv entirely.  Given w2c = [R | T], the analytic
    inverse is c2w = [R^T | -R^T @ T], which has bounded, well-conditioned
    gradients even at random initialisation.
    """
    B, S, _ = pose_enc.shape
    p = pose_enc.reshape(B * S, 9)
    T, quat = p[:, :3], p[:, 3:7]
    fov_h = p[:, 7].clamp(min=0.1, max=3.0)
    fov_w = p[:, 8].clamp(min=0.1, max=3.0)

    # Normalise quaternion so quat_to_mat always receives a unit quaternion.
    quat = F.normalize(quat, dim=-1)
    R = quat_to_mat(quat)                        # [B*S, 3, 3]

    R_T = R.transpose(1, 2)                      # [B*S, 3, 3]
    t_c = -torch.bmm(R_T, T.unsqueeze(-1))       # [B*S, 3, 1]
    Rt_c2w = torch.cat([R_T, t_c], dim=-1)       # [B*S, 3, 4]
    bottom = torch.tensor([[0, 0, 0, 1]], dtype=p.dtype, device=p.device
                          ).expand(B * S, -1, -1)
    c2w = torch.cat([Rt_c2w, bottom], dim=1)     # [B*S, 4, 4]

    H, W = image_hw
    fy = (H / 2.0) / torch.tan(fov_h / 2.0)
    fx = (W / 2.0) / torch.tan(fov_w / 2.0)
    fxfycxcy = torch.stack([fx / W, fy / H,
                            torch.full_like(fx, 0.5),
                            torch.full_like(fy, 0.5)], -1)
    return c2w, fxfycxcy


def _hierarchical_mask(levels, device):
    """levels [N] int -> [N,N] bool; mask[i,j] = level[j] <= level[i]."""
    return levels.unsqueeze(0) <= levels.unsqueeze(1)


# ---------------------------------------------------------------------------
# Spa3R main model
# ---------------------------------------------------------------------------

class Spa3R(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        D = config.model.embed_dim
        heads = config.model.num_heads
        agg_depth = config.model.aggregator_depth
        dec_depth = config.model.decoder_depth
        ps = config.model.patch_size
        img_sz = config.model.image_size
        n_reg = config.model.get("num_register_tokens", 4)
        mlp_r = config.model.get("mlp_ratio", 4.0)
        iv = config.model.get("init_values", 0.01)
        qk = config.model.get("qk_norm", True)
        rf = config.model.get("rope_freq", 100)

        self.embed_dim = D
        self.patch_size_val = ps
        self.img_size = img_sz
        self.h_patches = img_sz // ps
        self.w_patches = img_sz // ps
        self.n_patches = self.h_patches * self.w_patches
        self.num_register_tokens = n_reg
        self.num_pose_tokens = config.model.get("num_pose_tokens", 32)
        self.patch_start_idx = self.num_pose_tokens + n_reg

        # --- backbone ---
        self._build_patch_embed(config, n_reg)

        # --- RoPE ---
        self.rope = RotaryPositionEmbedding2D(frequency=rf) if rf > 0 else None
        self.position_getter = PositionGetter() if self.rope else None

        # --- aggregator ---
        self.frame_blocks = nn.ModuleList([
            MaskedBlock(D, heads, mlp_ratio=mlp_r, init_values=iv, qk_norm=qk, rope=self.rope)
            for _ in range(agg_depth)
        ])
        self.global_blocks = nn.ModuleList([
            MaskedBlock(D, heads, mlp_ratio=mlp_r, init_values=iv, qk_norm=qk, rope=self.rope)
            for _ in range(agg_depth)
        ])
        self.agg_depth = agg_depth

        # --- camera / pose tokens ---
        self.camera_token = nn.Parameter(torch.randn(1, 2, self.num_pose_tokens, D))
        self.register_token = nn.Parameter(torch.randn(1, 2, n_reg, D))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        self.register_buffer("_mean", torch.FloatTensor(_RESNET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.FloatTensor(_RESNET_STD).view(1, 3, 1, 1), persistent=False)

        # --- pose predictor ---
        self.pose_predictor = PosePredictor(dim=D)
        for p in self.pose_predictor.parameters():
            p.requires_grad_(False)

        # --- latent pose + renderer ---
        pose_dim = config.model.get("pose_dim", 256)
        self.pose_post_norm = nn.LayerNorm(D)
        self.latent_pose_head = LatentPoseHead(D, pose_dim)
        self.render = LatentRenderer(D, heads, dec_depth, ps, pose_dim)

        # --- loss ---
        self.loss_computer = LossComputer(config)
        self._lpips_weight = config.training.get("lpips_loss_weight", 0.1)
        if self._lpips_weight > 0 and lpips_lib is not None:
            self._train_lpips = lpips_lib.LPIPS(net="vgg")
            self._train_lpips.eval()
            for p in self._train_lpips.parameters():
                p.requires_grad_(False)

        self.freeze_backbone = config.model.get("freeze_backbone", True)
        self._self_decode_prob = config.model.get("self_decode_prob", 0.05)
        self._full_chance = config.model.get("full_chance", 0.05)
        self._empty_chance = config.model.get("empty_chance", 0.0)
        if self.freeze_backbone:
            for p in self.patch_embed.parameters():
                p.requires_grad_(False)

        self.config_bk = copy.deepcopy(config)

    # ---- backbone builder ----
    def _build_patch_embed(self, config, n_reg):
        pet = config.model.get("patch_embed", "conv")
        if pet == "conv":
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size_val,
                in_chans=3, embed_dim=self.embed_dim,
            )
        else:
            vit_map = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }
            self.patch_embed = vit_map[pet](
                img_size=518, patch_size=self.patch_size_val,
                num_register_tokens=n_reg,
                block_chunks=0,
                init_values=1e-5,
            )
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)
        bp = config.model.get("backbone_pretrained", None)
        if bp:
            self._load_backbone(bp)

    def _load_backbone(self, path):
        if path == "auto":
            name = self.config.model.get("patch_embed", "dinov2_vitl14_reg")
            print(f"[Spa3R] Loading backbone via torch.hub: {name}")
            m = torch.hub.load("facebookresearch/dinov2", name)
            sd = m.state_dict()
        else:
            print(f"[Spa3R] Loading backbone from: {path}")
            sd = torch.load(path, map_location="cpu", weights_only=True)
            if "model" in sd:
                sd = sd["model"]
        miss, unex = self.patch_embed.load_state_dict(sd, strict=False)
        print(f"[Spa3R] Backbone loaded — missing {len(miss)}, unexpected {len(unex)}")

    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()
        if hasattr(self, '_train_lpips'):
            self._train_lpips.eval()

    # ---- helpers ----
    def _get_pos(self, BS, device):
        if self.rope is None:
            return None
        pos = self.position_getter(BS, self.h_patches, self.w_patches, device) + 1
        sp = torch.zeros(BS, self.patch_start_idx, 2, device=device, dtype=pos.dtype)
        return torch.cat([sp, pos], dim=1)

    def _get_pos_tgt_train(self, B, device):
        """Target frame has an extra pose_token_2 → one more special-token slot."""
        if self.rope is None:
            return None
        pos = self.position_getter(B, self.h_patches, self.w_patches, device) + 1
        n_special = 2 * self.num_pose_tokens + self.num_register_tokens
        sp = torch.zeros(B, n_special, 2, device=device, dtype=pos.dtype)
        return torch.cat([sp, pos], dim=1)

    def _patchify(self, images):
        """images [B, 3, H, W] in [0,1] -> [B, N, C]."""
        images = (images - self._mean) / self._std
        if self.freeze_backbone:
            with torch.no_grad():
                tok = self.patch_embed(images)
        else:
            tok = self.patch_embed(images)
        if isinstance(tok, dict):
            tok = tok["x_norm_patchtokens"]
        return tok

    # ---- forward dispatch ----
    def forward(self, data, create_visual=False, render_video=False, iter=0):
        if self.training:
            return self._forward_train(data, create_visual=create_visual)
        return self._forward_inference(data, create_visual)

    # ================================================================
    #  TRAINING FORWARD
    # ================================================================
    def _forward_train(self, data, create_visual=False):
        imgs = data["image"]                       # [B, 2, 3, H, W]
        B, V, _, H, W = imgs.shape
        device = imgs.device

        smask = (torch.rand(B, device=device) <= self._self_decode_prob).float()
        imgs = imgs.clone()
        imgs[:, 1] = (1.0 - smask[:, None, None, None]) * imgs[:, 1] + smask[:, None, None, None] * imgs[:, 0]

        ctx_img = imgs[:, 0]                       # [B, 3, H, W]
        tgt_img = imgs[:, 1]

        # 1. patchify
        ctx_patches = self._patchify(ctx_img)      # [B, N, C]
        tgt_patches = self._patchify(tgt_img)
        N, C = ctx_patches.shape[1], ctx_patches.shape[2]

        # 2. quadrant mask
        pmask, qc = generate_quadrant_mask(
            B, self.h_patches, self.w_patches, device,
            self._full_chance, self._empty_chance,
        )  # [B,N] 0=a 1=b

        # 3. special tokens
        cam_ctx  = self.camera_token[:, 0].expand(B, -1, -1)       # [B,1,C]
        reg_ctx  = self.register_token[:, 0].expand(B, -1, -1)     # [B,R,C]
        cam_base = self.camera_token[:, 1].expand(B, -1, -1)       # [B,1,C]
        reg_tgt  = self.register_token[:, 1].expand(B, -1, -1)

        pose1 = cam_base.clone()
        pose2 = cam_base.clone()

        ctx_tok = torch.cat([cam_ctx, reg_ctx, ctx_patches], 1)    # [B, P_c, C]
        tgt_tok = torch.cat([pose1, pose2, reg_tgt, tgt_patches], 1)  # [B, P_t, C]
        P_c, P_t = ctx_tok.shape[1], tgt_tok.shape[1]

        # 4. visibility levels  (0: ctx, 1: aug_a, 2: aug_b / full)
        tgt_levels = torch.full((B, P_t), 2, dtype=torch.long, device=device)
        tgt_levels[:, :self.num_pose_tokens] = 1                   # pose_token_1 group
        patch_off = 2 * self.num_pose_tokens + self.num_register_tokens
        for b_i in range(B):
            tgt_levels[b_i, patch_off:][pmask[b_i] == 0] = 1       # aug_a patches

        ctx_levels = torch.zeros(P_c, dtype=torch.long, device=device)

        # 5. attention masks
        tgt_frame_mask = torch.stack([
            _hierarchical_mask(tgt_levels[b_i], device) for b_i in range(B)
        ]).unsqueeze(1)                                            # [B,1,P_t,P_t]

        gl_levels = torch.cat([
            ctx_levels.unsqueeze(0).expand(B, -1), tgt_levels
        ], 1)                                                      # [B, P_c+P_t]
        global_mask = torch.stack([
            _hierarchical_mask(gl_levels[b_i], device) for b_i in range(B)
        ]).unsqueeze(1)                                            # [B,1,T,T]

        # 6. positions
        pos_ctx = self._get_pos(B, device)
        pos_tgt = self._get_pos_tgt_train(B, device)

        # 7. aggregator (alternating frame / global)
        for i in range(self.agg_depth):
            ctx_tok = torch_checkpoint(
                self.frame_blocks[i], ctx_tok, pos_ctx, None,
                use_reentrant=False,
            )
            tgt_tok = torch_checkpoint(
                self.frame_blocks[i], tgt_tok, pos_tgt, tgt_frame_mask,
                use_reentrant=False,
            )

            all_tok = torch.cat([ctx_tok, tgt_tok], 1)
            gl_pos = torch.cat([pos_ctx, pos_tgt], 1) if pos_ctx is not None else None
            all_tok = torch_checkpoint(
                self.global_blocks[i], all_tok, gl_pos, global_mask,
                use_reentrant=False,
            )
            ctx_tok, tgt_tok = all_tok.split([P_c, P_t], 1)

        # 8. extract features
        cam_ctx_feat = ctx_tok[:, :self.num_pose_tokens]
        pose1_feat   = tgt_tok[:, :self.num_pose_tokens]
        pose2_feat   = tgt_tok[:, self.num_pose_tokens:2 * self.num_pose_tokens]
        ctx_patch_f  = ctx_tok[:, self.patch_start_idx:]           # [B,N,C]

        # 9. explicit pose head kept only for compatibility / pose eval
        pose_tok_all = torch.stack([cam_ctx_feat[:, 0], pose1_feat[:, 0]], dim=1)  # [B,2,C]
        with torch.no_grad():
            pose_enc1 = self.pose_predictor(pose_tok_all.detach())[-1]      # [B,2,9]
            c2w, fxfycxcy = pose_enc_to_c2w_fxfycxcy(pose_enc1, (H, W))

        # 10. xfactor-style latent pose conditioning for rendering
        pose_feats = torch.stack([
            torch.stack([cam_ctx_feat[:, 0], cam_ctx_feat[:, 0]], dim=1),
            torch.stack([pose1_feat[:, 0], pose2_feat[:, 0]], dim=1),
        ], dim=1)  # [B, 2, 2, C]
        pose_feats = F.gelu(self.pose_post_norm(pose_feats))
        pose_ref = pose_feats[:, 0:1].expand_as(pose_feats)
        latent_pose = self.latent_pose_head(pose_feats, pose_ref)  # [B,2,2,Cp]
        P0, P1 = latent_pose[:, :, 0], latent_pose[:, :, 1]
        q0 = (qc == 0).float()  # full_chance: target sees all quadrants
        q1 = (qc == 4).float()  # empty_chance: target sees no quadrants
        P0n = (1.0 - q1[:, None, None]) * P0 + q1[:, None, None] * P1
        P1n = (1.0 - q0[:, None, None]) * P1 + q0[:, None, None] * P0
        latent_pose_swap = torch.stack([P1n, P0n], dim=2)

        H_eff = self.h_patches * self.patch_size_val
        W_eff = self.w_patches * self.patch_size_val
        tgt_img_eff = tgt_img[:, :, :H_eff, :W_eff]
        pred_full = self.render(
            ctx_patch_f.unsqueeze(1),
            ctx_img[:, None, :, :H_eff, :W_eff],
            latent_pose_swap,
            pmask,
        )

        pred_rgb_patches = rearrange(
            pred_full,
            "b c (hh ph) (ww pw) -> b (hh ww) (ph pw c)",
            ph=self.patch_size_val, pw=self.patch_size_val,
        )
        tgt_rgb_patches = rearrange(
            tgt_img_eff,
            "b c (hh ph) (ww pw) -> b (hh ww) (ph pw c)",
            ph=self.patch_size_val, pw=self.patch_size_val,
        )

        aug_b_idx = (pmask == 1)                                   # [B,N]
        valid_counts = aug_b_idx.sum(dim=1)
        use_masked = valid_counts > 0

        # 11. losses: masked aug_b supervision when available, full-image fallback for full_chance samples
        smooth_l1_beta = self.config.training.get("smooth_l1_beta", 0.05)
        pred_rgb_flat = pred_rgb_patches.reshape(B, -1, pred_rgb_patches.shape[-1])
        tgt_rgb_flat = tgt_rgb_patches.reshape(B, -1, tgt_rgb_patches.shape[-1])

        per_sample_smooth_l1 = []
        per_sample_mse = []
        for b in range(B):
            if use_masked[b]:
                pred_sel = pred_rgb_flat[b][aug_b_idx[b]]
                tgt_sel = tgt_rgb_flat[b][aug_b_idx[b]]
            else:
                pred_sel = pred_rgb_flat[b]
                tgt_sel = tgt_rgb_flat[b]
            per_sample_smooth_l1.append(F.smooth_l1_loss(pred_sel, tgt_sel, beta=smooth_l1_beta))
            with torch.no_grad():
                per_sample_mse.append(F.mse_loss(pred_sel, tgt_sel))

        smooth_l1 = torch.stack(per_sample_smooth_l1).mean()
        mse_for_psnr = torch.stack(per_sample_mse).mean()
        psnr = -10.0 * torch.log10(mse_for_psnr.clamp(min=1e-8))

        loss = smooth_l1
        lpips_val = torch.tensor(0.0, device=device)

        mask_f = aug_b_idx.unsqueeze(-1).float()
        assembled = tgt_rgb_patches.detach() * (1 - mask_f) + pred_rgb_patches * mask_f
        pred_img = rearrange(
            assembled,
            "b (hh ww) (ph pw c) -> b c (hh ph) (ww pw)",
            hh=self.h_patches, ww=self.w_patches,
            ph=self.patch_size_val, pw=self.patch_size_val, c=3,
        )

        if self._lpips_weight > 0 and hasattr(self, '_train_lpips'):
            lpips_val = self._train_lpips(
                (pred_img * 2.0 - 1.0).float(),
                (tgt_img_eff * 2.0 - 1.0).float(),
            ).mean()
            loss = loss + self._lpips_weight * lpips_val

        ret = edict(
            loss_metrics=edict(
                loss=loss, smooth_l1_loss=smooth_l1,
                lpips_loss=lpips_val, psnr=psnr,
            ),
            c2w=c2w.view(B, 2, 4, 4),
            fxfycxcy=fxfycxcy.view(B, 2, 4),
        )

        if create_visual:
            ctx_img_eff = ctx_img[:, :, :H_eff, :W_eff]
            ret.input = edict(
                image=ctx_img_eff.unsqueeze(1).detach().clone(),
                index=data["index"][:, 0:1],
                scene_name=data.get("scene_name", [""]*B),
            )
            ret.target = edict(
                image=tgt_img_eff.unsqueeze(1).detach().clone(),
                index=data["index"][:, 1:2],
                scene_name=data.get("scene_name", [""]*B),
            )
            ret.render = pred_img.unsqueeze(1).detach()

        return ret

    # ================================================================
    #  INFERENCE FORWARD
    # ================================================================
    def _forward_inference(self, data, create_visual=False):
        imgs = data["image"]                                       # [B, V, 3, H, W]
        B, V, _, H, W = imgs.shape
        device = imgs.device
        num_in = self.config.training.num_input_views

        # patchify all views
        all_patches = self._patchify(imgs.reshape(B * V, 3, H, W)).view(B, V, self.n_patches, self.embed_dim)

        # tokens per frame: [cam, reg, patches]
        cam0 = self.camera_token[:, 0].expand(B, -1, -1)
        cam1 = self.camera_token[:, 1].expand(B, -1, -1)
        reg0 = self.register_token[:, 0].expand(B, -1, -1)
        reg1 = self.register_token[:, 1].expand(B, -1, -1)

        ftoks = []
        for v in range(V):
            c = cam0 if v == 0 else cam1
            r = reg0 if v == 0 else reg1
            ftoks.append(torch.cat([c, r, all_patches[:, v]], 1))
        P = ftoks[0].shape[1]

        pos = self._get_pos(B, device)

        # aggregator – no mask
        for i in range(self.agg_depth):
            for v in range(V):
                ftoks[v] = self.frame_blocks[i](ftoks[v], pos=pos)
            gl = torch.cat(ftoks, 1)
            gl_pos = pos.repeat(1, V, 1) if pos is not None else None
            gl = self.global_blocks[i](gl, pos=gl_pos)
            ftoks = list(gl.split(P, 1))

        # pose
        cam_feats = torch.stack([f[:, 0] for f in ftoks], 1)      # [B,V,C]
        with torch.no_grad():
            pose_enc = self.pose_predictor(cam_feats.detach())[-1]          # [B,V,9]
            c2w, fxfycxcy = pose_enc_to_c2w_fxfycxcy(pose_enc, (H, W))

        latent_feats = F.gelu(self.pose_post_norm(cam_feats))
        latent_ref = latent_feats[:, 0:1].expand_as(latent_feats)
        latent_pose = self.latent_pose_head(latent_feats, latent_ref)  # [B,V,Cp]

        H_eff = self.h_patches * self.patch_size_val
        W_eff = self.w_patches * self.patch_size_val
        ctx_feat = torch.stack([ftoks[v][:, self.patch_start_idx:] for v in range(num_in)], dim=1)
        x_ctx = imgs[:, :num_in, :, :H_eff, :W_eff]

        # render targets
        rendered = []
        for tv in range(num_in, V):
            P_v = torch.cat([latent_pose[:, :num_in], latent_pose[:, tv:tv + 1]], dim=1)
            img = self.render(ctx_feat, x_ctx, P_v, pmask=None)
            rendered.append(img)

        rendered = torch.stack(rendered, 1)                        # [B, V_t, 3, H', W']
        tgt_gt = data["image"][:, num_in:, :, :H_eff, :W_eff]

        loss_m = self.loss_computer(rendered, tgt_gt)

        input_idx  = torch.arange(num_in, device=device).unsqueeze(0).expand(B, -1)
        target_idx = torch.arange(num_in, V, device=device).unsqueeze(0).expand(B, -1)

        inp_dict = edict(image=data["image"][:, :num_in, :, :H_eff, :W_eff])
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
            c2w=c2w.view(B, V, 4, 4),
            fxfycxcy=fxfycxcy.view(B, V, 4),
            input=inp_dict,
            target=tgt_dict,
            input_idx=input_idx,
            target_idx=target_idx,
        )

    # ---- checkpoint loading ----
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
            patch_embed=c(self.patch_embed),
            frame_blocks=c(self.frame_blocks),
            global_blocks=c(self.global_blocks),
            pose_predictor=c(self.pose_predictor),
            latent_pose_head=c(self.latent_pose_head),
            render=c(self.render),
        )
