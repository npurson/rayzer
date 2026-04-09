import copy

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from .transformer import QK_Norm_TransformerBlock, _init_weights_layerwise
from .transformer import init_weights as _init_weights
from .loss import LossComputer
from .gaussians_renderer import get_point_range_func, Renderer, build_stepback_c2ws

from utils.pe_utils import get_2d_sincos_pos_embed
from utils.pose_utils import rot6d2mat, quat2mat
from utils import camera_utils


def build_transformer_blocks(
    num_layers: int,
    d: int,
    d_head: int,
    use_qk_norm: bool,
    special_init: bool = False,
    depth_init: bool = False,
) -> nn.ModuleList:
    layers = [
        QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
        for _ in range(num_layers)
    ]
    if special_init:
        for idx, layer in enumerate(layers):
            if depth_init:
                std = 0.02 / (2 * (idx + 1)) ** 0.5
            else:
                std = 0.02 / (2 * num_layers) ** 0.5
            layer.apply(lambda module: _init_weights_layerwise(module, std))

    return nn.ModuleList(layers)


class GaussiansUpsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scaling_bias = self.config.model.get("scaling_bias", -2.3)
        self.scaling_max = self.config.model.get("scaling_max", -1.2)
        self.opacity_bias = self.config.model.get("opacity_bias", -2.0)

    def to_gs(self, gaussians):
        """
        Split raw Gaussian attributes and apply activation/bias.
        Args:
            gaussians: [b, n_gaussians, d]
        Returns:
            xyz, features, scaling, rotation, opacity
        """
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=2
        )

        if not self.config.model.hard_pixelalign:
            xyz = xyz.clamp(-500.0, 500.0)

        features = features.reshape(
            features.size(0), features.size(1),
            (self.config.model.gaussians.sh_degree + 1) ** 2, 3,
        )

        scaling = (scaling + self.scaling_bias).clamp(max=self.scaling_max).clamp(min=-10.0)
        opacity = (opacity + self.opacity_bias).clamp(min=-10.0)

        return xyz, features, scaling, rotation, opacity


class ERayZer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d = self.config.model.transformer.d
        self.d_head = self.config.model.transformer.d_head
        self.hh = self.ww = (
            self.config.model.image_tokenizer.image_size
            // self.config.model.image_tokenizer.patch_size
        )
        self.ph = self.pw = self.config.model.image_tokenizer.patch_size

        # Image tokenizer: patches -> tokens
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.ph, pw=self.pw,
            ),
            nn.Linear(
                self.config.model.image_tokenizer.in_channels * (self.ph * self.pw),
                self.d, bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # Spatial PE embedders (image and Plücker ray)
        self.use_pe_embedding_layer = self.config.model.get("input_with_pe", True)
        if self.use_pe_embedding_layer:
            self.pe_embedder = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.SiLU(),
                nn.Linear(self.d, self.d),
            )
            self.pe_embedder.apply(_init_weights)

            self.pe_embedder_plucker = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.SiLU(),
                nn.Linear(self.d, self.d),
            )
            self.pe_embedder_plucker.apply(_init_weights)

        # VGGT-style camera token and register tokens
        self.num_register_tokens = 4
        self.camera_token = nn.Parameter(torch.randn(1, 1, self.d))
        self.register_token = nn.Parameter(torch.randn(1, self.num_register_tokens, self.d))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        use_qk_norm = self.config.model.transformer.get("use_qk_norm", False)

        # Pose encoder (alternating frame/global attention)
        self.transformer_encoder = build_transformer_blocks(
            num_layers=config.model.transformer.encoder_n_layer,
            d=self.d, d_head=self.d_head, use_qk_norm=use_qk_norm,
            special_init=config.model.transformer.get("special_init", False),
            depth_init=config.model.transformer.get("depth_init", False),
        )

        # Geometry encoder (alternating frame/global attention)
        self.transformer_encoder_geom = build_transformer_blocks(
            num_layers=config.model.transformer.encoder_geom_n_layer,
            d=self.d, d_head=self.d_head, use_qk_norm=use_qk_norm,
            special_init=config.model.transformer.get("special_init", False),
            depth_init=config.model.transformer.get("depth_init", False),
        )

        # Pose predictor
        self.pose_predictor = PoseEstimator(self.config)

        # Input Plücker ray tokenizer
        self.input_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v (hh ph) (ww pw) c -> (b v) (hh ww) (ph pw c)",
                ph=self.ph, pw=self.pw,
            ),
            nn.Linear(6 * (self.ph * self.pw), self.d, bias=False),
        )
        self.input_pose_tokenizer.apply(_init_weights)

        # Image-ray fusion MLP
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(self.d * 2, bias=False),
            nn.Linear(self.d * 2, self.d, bias=True),
            nn.SiLU(),
            nn.Linear(self.d, self.d, bias=True),
        )
        self.mlp_fuse.apply(_init_weights)

        # 3D Gaussian decoder: token -> per-pixel Gaussian attributes
        gs_attr_dim = 3 + (self.config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.d, bias=False),
            nn.Linear(self.d, (self.ph * self.pw) * gs_attr_dim, bias=False),
        )
        self.image_token_decoder.apply(_init_weights)

        self.upsampler = GaussiansUpsampler(self.config)
        self.range_func = get_point_range_func(self.config.model.gaussians)
        self.renderer = Renderer(self.config)

        # Loss
        self.loss_computer = LossComputer(config)

        self.config_bk = copy.deepcopy(self.config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        if config.model.transformer.get("fix_decoder", False):
            self.freeze_weights()

        if config.inference or config.get("evaluation", False):
            self.random_index = config.training.get("random_inputs", False)
        else:
            self.random_index = config.training.get("random_split", False)
        print("Use random index:", self.random_index)

    def train(self, mode=True):
        super().train(mode)
        self.loss_computer.eval()

    def forward(self, data, create_visual=False, render_video=False, iter=0):
        image_all = data["image"] * 2.0 - 1.0  # [b, v_all, c, h, w]
        b, v, c, h, w = image_all.shape
        device = image_all.device

        # Pad to 10 views if needed (repeat last view)
        if v < 10:
            pad_input = True
            v_all = 10
            pad_views = 10 - v
            last_view = image_all[:, -1:, ...].repeat(1, pad_views, 1, 1, 1)
            image_all = torch.cat([image_all, last_view], dim=1)
        else:
            pad_input = False
            v_all = v

        # === Stage 1: SE(3) pose prediction for all views ===
        img_tokens = self.image_tokenizer(image_all)  # [b*v_all, n, d]
        _, n, d = img_tokens.shape

        if self.use_pe_embedding_layer:
            img_tokens = self.add_spatial_pe(
                img_tokens, b, v_all, self.hh, self.ww, embedder=self.pe_embedder,
            )

        cam_tokens = repeat(self.camera_token, "1 n d -> bv n d", bv=b * v_all)
        register_tokens = repeat(self.register_token, "1 n d -> bv n d", bv=b * v_all)
        all_tokens = torch.cat([cam_tokens, register_tokens, img_tokens], dim=1)
        _, n2, _ = all_tokens.shape
        all_tokens = rearrange(all_tokens, "(b v) n d -> b (v n) d", b=b)

        # Alternating frame/global attention encoder
        all_tokens = self.run_vggt_encoder(all_tokens, b, v_all)
        all_tokens = rearrange(all_tokens, "b (v n) d -> (b v) n d", v=v_all)
        cam_tokens, _, _ = all_tokens.split([1, self.num_register_tokens, n], dim=1)

        # Predict SE(3) poses and intrinsics
        cam_tokens = cam_tokens[:, 0]  # [b*v_all, d]
        cam_info = self.pose_predictor(cam_tokens, v_all)
        pred_c2w, pred_fxfycxcy = get_cam_se3(cam_info)
        pred_c2w = rearrange(pred_c2w, "(b v) n d -> b v n d", b=b)
        pred_fxfycxcy = rearrange(pred_fxfycxcy, "(b v) d -> b v d", b=b).detach()
        normalized = True

        # === Stage 2: Plücker ray embeddings ===
        if v < 5:
            v_input = 5
            c2w_input = pred_c2w[:, :5, ...]
            fxfycxcy_input = pred_fxfycxcy[:, :5, ...]
            img_tokens_input = rearrange(img_tokens, "(b v) n d -> b v n d", b=b)[:, :5, ...]
        else:
            v_input = v
            c2w_input = pred_c2w[:, :v, ...]
            fxfycxcy_input = pred_fxfycxcy[:, :v, ...]
            img_tokens_input = rearrange(img_tokens, "(b v) n d -> b v n d", b=b)[:, :v, ...]

        c2w_target = pred_c2w[:, :v]
        fxfycxcy_target = pred_fxfycxcy[:, :v]

        plucker_rays_input = cam_info_to_plucker(
            c2w_input, fxfycxcy_input, self.config.model.target_image,
            normalized=normalized, return_moment=True,
        )
        plucker_rays_input = rearrange(plucker_rays_input, "(b v) c h w -> b v h w c", b=b, v=v_input)
        plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)
        if self.use_pe_embedding_layer:
            plucker_emb_input = self.add_spatial_pe(
                plucker_emb_input, b, v_input, self.hh, self.ww,
                embedder=self.pe_embedder_plucker,
            )
        plucker_emb_input = rearrange(plucker_emb_input, "(b v) n d -> b (v n) d", v=v_input)

        # === Stage 3: Scene representation (geometry encoder) ===
        img_tokens_input = rearrange(img_tokens_input, "b v n d -> b (v n) d")
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)
        all_tokens = self.mlp_fuse(img_tokens_input)

        all_tokens = self.run_vggt_encoder_geom(all_tokens, b, v_input)

        # === Stage 4: Gaussian prediction ===
        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians, "b (v n) d -> b v n d", v=v_input,
        )[:, :v]
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            "b v n (ph pw c) -> b (v n ph pw) c",
            ph=self.ph, pw=self.pw,
        )
        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)

        # Hard pixel alignment: convert depth to 3D positions along rays
        img_aligned_xyz = rearrange(
            xyz,
            "b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)",
            v=v, hh=self.hh, ww=self.ww, ph=self.ph, pw=self.pw,
        )

        if self.config.model.hard_pixelalign:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays_raw = cam_info_to_plucker(
                c2w_input[:, :v, ...], fxfycxcy_input[:, :v, ...],
                self.config.model.target_image, normalized=normalized, return_moment=False,
            )
            plucker_rays_raw = rearrange(plucker_rays_raw, "(b v) c h w -> b v c h w", b=b)
            ray_o, ray_d = plucker_rays_raw.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            xyz = rearrange(
                img_aligned_xyz,
                "b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c",
                ph=self.ph, pw=self.pw,
            )

        # === Stage 5: Render target views ===
        height, width = h, w
        if normalized:
            fxfycxcy_target_render = fxfycxcy_target.clone()
            fxfycxcy_target_render[..., 0] *= width
            fxfycxcy_target_render[..., 1] *= height
            fxfycxcy_target_render[..., 2] *= width
            fxfycxcy_target_render[..., 3] *= height
        else:
            fxfycxcy_target_render = fxfycxcy_target

        render = self.renderer(
            xyz, features, scaling, rotation, opacity,
            height, width, C2W=c2w_target, fxfycxcy=fxfycxcy_target_render,
        )
        rendered_images = render.render  # [b, v, 3, h, w]

        # === Stage 6: Compute loss ===
        rendered_clamped = rendered_images.clamp(0, 1)
        target_images = data["image"][:, :v]  # [b, v, c, h, w], range [0, 1]
        loss_metrics = self.loss_computer(rendered_clamped, target_images)

        # Build result dict
        result = edict(
            loss_metrics=loss_metrics,
            render=rendered_clamped,
            image=data["image"],
            c2w=pred_c2w,
            fxfycxcy=rearrange(pred_fxfycxcy, "b v d -> (b v) d"),
            c2w_input=c2w_input,
            fxfycxcy_input=fxfycxcy_input,
            c2w_target=c2w_target,
            fxfycxcy_target=fxfycxcy_target,
        )

        # Optional: visualization video
        if create_visual and render_video:
            gaussian_attrs = edict(
                xyz=xyz, features=features, scaling=scaling,
                rotation=rotation, opacity=opacity,
            )
            with torch.no_grad():
                vis_results = self.render_images_video(
                    gaussian_attrs, c2w_target, fxfycxcy_target_render,
                    normalized=False,
                    step_back=self.config.get("evaluation_step_back_distance", 0),
                )
                result.video_rendering = vis_results.rendered_images_video.detach().clamp(0, 1)

        return result

    def add_spatial_pe(self, tokens, b, v, h_tokens, w_tokens, embedder):
        """Add 2D sinusoidal spatial positional encoding to tokens."""
        bv, n, d = tokens.shape
        assert (h_tokens * w_tokens) == n, f"Token count {n} != h*w {h_tokens}x{w_tokens}"

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d, grid_size=(h_tokens, w_tokens), device=tokens.device,
        ).to(tokens.dtype)

        spatial_pe = spatial_pe.reshape(1, 1, n, d).repeat(b, v, 1, 1)
        spatial_pe = spatial_pe.reshape(bv, n, d)
        pe = embedder(spatial_pe)
        return tokens + pe

    def run_vggt_encoder(self, all_tokens_encoder, b, v):
        """Run pose encoder with alternating frame/global attention."""
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            if i % 2 == 0:
                all_tokens_encoder = rearrange(
                    all_tokens_encoder, "b (v n) d -> (b v) n d", v=v
                )
            else:
                all_tokens_encoder = rearrange(
                    all_tokens_encoder, "(b v) n d -> b (v n) d", b=b
                )

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self._run_layers_encoder(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self._run_layers_encoder(
                    i + 1, i + checkpoint_every
                )(all_tokens_encoder)
        return all_tokens_encoder

    def run_vggt_encoder_geom(self, all_tokens_encoder, b, v):
        """Run geometry encoder with alternating frame/global attention."""
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            if i % 2 == 0:
                all_tokens_encoder = rearrange(
                    all_tokens_encoder, "b (v n) d -> (b v) n d", v=v
                )
            else:
                all_tokens_encoder = rearrange(
                    all_tokens_encoder, "(b v) n d -> b (v n) d", b=b
                )

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self._run_layers_encoder_geom(i, i + 1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self._run_layers_encoder_geom(
                    i + 1, i + checkpoint_every
                )(all_tokens_encoder)
        return all_tokens_encoder

    def _run_layers_encoder(self, start, end):
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                tokens = self.transformer_encoder[i](tokens)
            return tokens
        return custom_forward

    def _run_layers_encoder_geom(self, start, end):
        def custom_forward(tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                tokens = self.transformer_encoder_geom[i](tokens)
            return tokens
        return custom_forward

    def render_images_video(self, gaussian_attrs, c2w_all, fxfycxcy_all, normalized=False, step_back=0):
        """Render interpolated video from Gaussians for visualization."""
        with torch.no_grad():
            xyz = gaussian_attrs.xyz.detach()
            features = gaussian_attrs.features.detach()
            scaling = gaussian_attrs.scaling.detach()
            rotation = gaussian_attrs.rotation.detach()
            opacity = gaussian_attrs.opacity.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, v, _, _ = c2w_all.shape
            device = xyz.device
            all_renderings = []
            num_frames = 30

            for i in range(b):
                c2ws = c2w_all[i]
                fxfycxcy = fxfycxcy_all[i]

                Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                Ks[:, 0, 0] = fxfycxcy[:, 0]
                Ks[:, 1, 1] = fxfycxcy[:, 1]
                Ks[:, 0, 2] = fxfycxcy[:, 2]
                Ks[:, 1, 2] = fxfycxcy[:, 3]
                c2ws_interp, Ks_interp = camera_utils.get_interpolated_poses_many(
                    c2ws[:, :3, :4], Ks, num_frames, order_poses=False,
                )
                frame_c2ws = torch.cat([
                    c2ws_interp.to(device),
                    torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2ws_interp.shape[0], 1, 1),
                ], dim=1)
                frame_fxfycxcy = torch.zeros((c2ws_interp.shape[0], 4), device=device)
                frame_fxfycxcy[:, 0] = Ks_interp[:, 0, 0]
                frame_fxfycxcy[:, 1] = Ks_interp[:, 1, 1]
                frame_fxfycxcy[:, 2] = Ks_interp[:, 0, 2]
                frame_fxfycxcy[:, 3] = Ks_interp[:, 1, 2]

                if step_back > 0:
                    frame_c2ws = build_stepback_c2ws(frame_c2ws, step_back_distance=step_back)

                batch_size = 5
                num_views = frame_c2ws.shape[0]
                renderings = []
                for start in range(0, num_views, batch_size):
                    end = min(start + batch_size, num_views)
                    batch_c2w = frame_c2ws[start:end].unsqueeze(0)
                    batch_fx = frame_fxfycxcy[start:end].unsqueeze(0)
                    rendered = self.renderer(
                        xyz, features, scaling, rotation, opacity,
                        self.config.model.image_tokenizer.image_size,
                        self.config.model.image_tokenizer.image_size,
                        C2W=batch_c2w, fxfycxcy=batch_fx,
                    ).render.squeeze(0)
                    renderings.append(rendered)
                all_renderings.append(torch.cat(renderings, dim=0))

            all_renderings = torch.stack(all_renderings)

        return edict(rendered_images_video=all_renderings)


# ============ Helper functions ============

def get_cam_se3(cam_info):
    """Convert camera info to SE(3) pose and intrinsics."""
    b, n = cam_info.shape
    if n == 13:
        rot_6d = cam_info[:, :6]
        R = rot6d2mat(rot_6d)
        t = cam_info[:, 6:9].unsqueeze(-1)
        fxfycxcy = cam_info[:, 9:]
    elif n == 11:
        rot_quat = cam_info[:, :4]
        R = quat2mat(rot_quat)
        t = cam_info[:, 4:7].unsqueeze(-1)
        fxfycxcy = cam_info[:, 7:]
    else:
        raise NotImplementedError(f"Unsupported cam_info dimension: {n}")

    Rt = torch.cat([R, t], dim=2)
    bottom = torch.tensor(
        [0, 0, 0, 1], dtype=R.dtype, device=R.device
    ).view(1, 1, 4).repeat(b, 1, 1)
    c2w = torch.cat([Rt, bottom], dim=1)
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True, return_moment=True):
    """
    Convert camera poses to Plücker ray coordinates.
    Args:
        c2w: [b, 4, 4] or [b, v, 4, 4]
        fxfycxcy: [b, 4] or [b, v, 4]
        return_moment: if True, return (moment, direction); otherwise (origin, direction)
    """
    if len(c2w.shape) == 3:
        b = c2w.shape[0]
    elif len(c2w.shape) == 4:
        c2w = rearrange(c2w.clone(), "b v n d -> (b v) n d")
        fxfycxcy = rearrange(fxfycxcy.clone(), "b v d -> (b v) d")
        b = c2w.shape[0]

    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)

    if return_moment:
        plucker = torch.cat([torch.cross(ray_o, ray_d, dim=1), ray_d], dim=1)
    else:
        plucker = torch.cat([ray_o, ray_d], dim=1)
    return plucker


# ============ Pose Estimator ============

class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.canonical = self.config.model.pose_latent.get("canonical", "first")
        assert self.canonical in ["first", "middle", "unordered"]
        self.is_pairwise = self.config.model.pose_latent.get("mode", "pairwise") == "pairwise"
        self.rel_head_input = (
            self.config.model.transformer.d * 2 if self.is_pairwise
            else config.model.transformer.d
        )

        self.pose_rep = self.config.model.pose_latent.get("representation", "6d")
        print("Pose representation:", self.pose_rep)
        if self.pose_rep == "6d":
            self.num_pose_element = 6
        elif self.pose_rep == "quat":
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        self.rel_head = nn.Sequential(
            nn.Linear(self.rel_head_input, config.model.transformer.d, bias=True),
            nn.SiLU(),
            nn.Linear(config.model.transformer.d, self.num_pose_element + 3, bias=True),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(config.model.transformer.d, config.model.transformer.d, bias=True),
            nn.SiLU(),
            nn.Linear(config.model.transformer.d, 1, bias=False),
        )
        self.canonical_k_head.apply(_init_weights)

        self.f_bias = 1.25

    def forward(self, x, v):
        if x.ndim == 2:
            x = rearrange(x, "(b v) d -> b v d", v=v)
            return_dim2 = True
        else:
            return_dim2 = False

        if self.is_pairwise:
            if v == 1:
                return self._forward_canonical_single(x, v, return_dim2)
            else:
                return self._forward_canonical(x, v, return_dim2)
        else:
            return self._forward_global(x, v, return_dim2)

    def _forward_global(self, x, v, return_dim2):
        """All-view global pose prediction (no canonical reference)."""
        b = x.shape[0]

        if self.pose_rep == "6d":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 0], device=x.device
            ).reshape(1, 1, 9).repeat(b, v, 1)
        elif self.pose_rep == "quat":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 0, 0, 0], device=x.device
            ).reshape(1, 1, 7).repeat(b, v, 1)
        else:
            raise NotImplementedError

        extrinsics_offset = self.rel_head(x)

        if self.canonical == "first":
            cano_idx = 0
        elif self.canonical == "middle":
            cano_idx = v // 2
        elif self.canonical == "unordered":
            cano_idx = None
        else:
            raise ValueError(f"Unknown canonical mode: {self.canonical}")

        if cano_idx is None:
            rt_final = rt_canonical + extrinsics_offset
        else:
            mask = torch.ones((1, v, 1), device=x.device)
            mask[:, cano_idx, :] = 0.0
            rt_final = rt_canonical + extrinsics_offset * mask

        fxfy_per_view = self.canonical_k_head(x) + self.f_bias
        fxfy_per_view = fxfy_per_view.repeat(1, 1, 2)
        fxfy_avg = fxfy_per_view.mean(dim=1, keepdim=True)
        fxfy_all = fxfy_avg.repeat(1, v, 1)

        info_all = torch.cat([rt_final, fxfy_all], dim=-1)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, "b v d -> (b v) d") if return_dim2 else info_all

    def _forward_canonical(self, x, v, return_dim2):
        """Canonical-frame-based pairwise pose prediction."""
        b = x.shape[0]
        canonical = self.canonical

        if canonical == "first":
            x_canonical = x[:, 0:1]
            x_rel = x[:, 1:]
        elif canonical == "middle":
            cano_idx = v // 2
            rel_indices = torch.cat([
                torch.arange(cano_idx), torch.arange(cano_idx + 1, v)
            ], dim=0).to(x.device)
            x_canonical = x[:, cano_idx:cano_idx + 1]
            x_rel = x[:, rel_indices]
        else:
            raise NotImplementedError

        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)

        if self.pose_rep == "6d":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device
            ).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == "quat":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device
            ).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)

        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)
        info_rel = self.rel_head(feat_rel)
        info_all = info_canonical.repeat(1, v, 1)

        if canonical == "first":
            info_all[:, 1:, :self.num_pose_element + 3] += info_rel
        elif canonical == "middle":
            info_all[:, rel_indices, :self.num_pose_element + 3] += info_rel

        cxcy_all = torch.tensor(
            [0.5, 0.5], device=info_all.device
        ).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, "b v d -> (b v) d") if return_dim2 else info_all

    def _forward_canonical_single(self, x, v, return_dim2):
        """Handle single-view case."""
        b = x.shape[0]
        x_canonical = x[:, 0:1]

        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)

        if self.pose_rep == "6d":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device
            ).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == "quat":
            rt_canonical = torch.tensor(
                [1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device
            ).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)
        info_all = info_canonical.repeat(1, v, 1)

        cxcy_all = torch.tensor(
            [0.5, 0.5], device=info_all.device
        ).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, "b v d -> (b v) d") if return_dim2 else info_all
