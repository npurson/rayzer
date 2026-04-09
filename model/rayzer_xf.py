import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
from einops import rearrange

from .rayzer import RayZer, get_cam_se3, cam_info_to_plucker


def generate_quadrant_mask(
    batch_size,
    h_patches,
    w_patches,
    device,
    full_chance=0.05,
    empty_chance=0.0,
):
    """Returns (pmask, qc), pmask shape [B, N] with values in {0, 1}."""
    assert h_patches % 2 == 0 and w_patches % 2 == 0
    if (
        full_chance < 0.0
        or empty_chance < 0.0
        or full_chance + empty_chance > 1.0
    ):
        raise ValueError(
            f"Expected 0 <= full_chance + empty_chance <= 1, got "
            f"full_chance={full_chance}, empty_chance={empty_chance}"
        )

    h2, w2 = h_patches // 2, w_patches // 2
    q = torch.zeros(batch_size, 4, dtype=torch.long, device=device)
    q[:, :2] = 1
    mask_mode = torch.rand(batch_size, device=device)
    full = mask_mode <= full_chance
    empty = (mask_mode > full_chance) & (mask_mode <= full_chance + empty_chance)
    q[full] = 0
    q[empty] = 1
    for i in range(batch_size):
        q[i] = q[i, torch.randperm(4, device=device)]

    qc = q.sum(1)
    q00 = q[:, 0].view(batch_size, 1, 1).expand(-1, h2, w2)
    q10 = q[:, 1].view(batch_size, 1, 1).expand(-1, h2, w2)
    q01 = q[:, 2].view(batch_size, 1, 1).expand(-1, h2, w2)
    q11 = q[:, 3].view(batch_size, 1, 1).expand(-1, h2, w2)
    top = torch.cat([q00, q10], dim=2)
    bot = torch.cat([q01, q11], dim=2)
    pmask = torch.cat([top, bot], dim=1).reshape(batch_size, -1)
    return pmask, qc


class RayZer_XF(RayZer):
    """RayZer with xFactor-style masked supervision and anti-leak strategy."""

    def __init__(self, config):
        super().__init__(config)
        self._self_decode_prob = config.model.get("self_decode_prob", 0.05)
        self._full_chance = config.model.get("full_chance", 0.05)
        self._empty_chance = config.model.get("empty_chance", 0.0)

    def render_images_grouped(self, scene_tokens, target_tokens, pmask_flat):
        """Segment-wise decoding without dense attention bias.

        For each sample, split target tokens into two segments by pmask and run the
        decoder independently for each segment together with the same scene tokens.
        This prevents inter-segment leakage and preserves flash attention.
        """
        b, _, _ = scene_tokens.shape
        bv = target_tokens.shape[0]
        v = bv // b
        scene_tokens = scene_tokens.unsqueeze(1).repeat(1, v, 1, 1)
        scene_tokens = rearrange(scene_tokens, "b v n d -> (b v) n d")
        n_target = target_tokens.shape[1]
        pred_patches = target_tokens.new_zeros(
            target_tokens.shape[0],
            target_tokens.shape[1],
            self.config.model.target_image.patch_size
            * self.config.model.target_image.patch_size
            * 3,
        )

        for i in range(bv):
            scene_i = scene_tokens[i : i + 1]  # [1, n_scene, d]
            for g in (0, 1):
                seg = pmask_flat[i] == g
                if not seg.any():
                    continue
                tgt_i = target_tokens[i : i + 1, seg]  # [1, n_seg, d]
                all_tokens = torch.cat([tgt_i, scene_i], dim=1)
                all_tokens = self.decoder_ln(all_tokens)
                all_tokens = self.run_decoder(all_tokens)
                tgt_out = all_tokens[:, : tgt_i.shape[1]]
                pred_seg = self.image_token_decoder(tgt_out).squeeze(0)
                pred_patches[i, seg] = pred_seg

        rendered_images_all = pred_patches
        patch_size = self.config.model.target_image.patch_size
        rendered_images_all = rearrange(
            rendered_images_all,
            "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
            v=v,
            h=self.target_latent_h,
            w=self.target_latent_w,
            p1=patch_size,
            p2=patch_size,
            c=3,
        )
        return edict(rendered_images=rendered_images_all)

    def forward(self, data, create_visual=False, render_video=False, iter=0):
        input, target, input_idx, target_idx = self.split_data(
            data, random_index=self.random_index
        )
        image = input.image * 2.0 - 1.0
        b, v_input, _, h, w = image.shape
        image_all = data["image"] * 2.0 - 1.0
        v_all = image_all.shape[1]
        v_target = v_all - v_input
        device = image.device
        input_idx, target_idx = input_idx.to(device), target_idx.to(device)
        batch_idx = torch.arange(b, device=device).unsqueeze(1)

        # xFactor self-decode: occasionally replace target views with reference input view.
        if self.training and v_target > 0 and self._self_decode_prob > 0.0:
            # Sample self-decode per target view (not per sample), avoiding
            # collapsing all target views to the same reference image.
            smask = (
                torch.rand(b, v_target, device=device) <= self._self_decode_prob
            ).float()
            ref_idx = input_idx[:, :1].expand(-1, v_target)
            source = image_all[batch_idx, ref_idx]
            tgt_orig = image_all[batch_idx, target_idx]
            image_all = image_all.clone()
            image_all[batch_idx, target_idx] = (
                (1.0 - smask[:, :, None, None, None]) * tgt_orig
                + smask[:, :, None, None, None] * source
            )

        # se3 pose prediction for all views
        img_tokens = self.image_tokenizer(image_all)
        _, n, _ = img_tokens.shape
        if self.use_pe_embedding_layer:
            img_tokens = self.add_sptial_temporal_pe(img_tokens, b, v_all, h, w)
        img_tokens = rearrange(img_tokens, "(b v) n d -> b (v n) d", b=b, v=v_all)

        cam_tokens = self.get_camera_tokens(b, v_all)
        n_cam = cam_tokens.shape[1] // v_all
        assert n_cam == 1
        cam_tokens = rearrange(cam_tokens, "b (v n) d -> b v n d", v=v_all)
        cam_tokens = rearrange(cam_tokens, "b v n d -> b (v n) d")

        all_tokens = torch.cat([cam_tokens, img_tokens], dim=1)
        all_tokens = self.run_encoder(all_tokens)
        cam_tokens, _ = all_tokens.split([v_all * n_cam, v_all * n], dim=1)

        cam_tokens = rearrange(cam_tokens, "b (v n) d -> (b v) n d", b=b, v=v_all, n=n_cam)[
            :, 0
        ]
        cam_info = self.pose_predictor(cam_tokens, v_all)
        c2w, fxfycxcy = get_cam_se3(cam_info)
        normalized = True

        plucker_rays = cam_info_to_plucker(
            c2w, fxfycxcy, self.config.model.target_image, normalized=normalized
        )
        plucker_rays = rearrange(plucker_rays, "(b v) c h w -> b v c h w", b=b, v=v_all)
        plucker_emb_input = self.target_pose_tokenizer(plucker_rays[batch_idx, input_idx])
        plucker_emb_target = self.target_pose_tokenizer2(plucker_rays[batch_idx, target_idx])
        plucker_emb_input = rearrange(
            plucker_emb_input, "(b v) n d -> b (v n) d", v=v_input
        )

        # scene tokens from input views
        img_tokens_input = rearrange(img_tokens, "b (v n) d -> b v n d", v=v_all)[
            batch_idx, input_idx
        ]
        img_tokens_input = rearrange(img_tokens_input, "b v n d -> b (v n) d")
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)
        img_tokens_input = self.mlp_fuse(img_tokens_input)

        scene_tokens = self.scene_code.expand(b, -1, -1)
        n_scene = scene_tokens.shape[1]
        all_tokens = torch.cat([scene_tokens, img_tokens_input], dim=1)
        all_tokens = self.run_encoder_geom(all_tokens)
        scene_tokens, _ = all_tokens.split([n_scene, v_input * n], dim=1)

        ph = self.config.model.target_image.patch_size
        hp = self.config.model.target_image.height // ph
        wp = self.config.model.target_image.width // ph
        num_patches = hp * wp
        pmask, _ = generate_quadrant_mask(
            b, hp, wp, device, self._full_chance, self._empty_chance
        )
        pmask_v = pmask.unsqueeze(1).expand(-1, v_target, -1).reshape(-1, num_patches)

        # render target images with pmask-grouped decoder attention
        render_results = self.render_images_grouped(
            scene_tokens, plucker_emb_target, pmask_flat=pmask_v
        )

        if create_visual and render_video:
            with torch.no_grad():
                c2w_target = rearrange(c2w, "(b v) c d -> b v c d", v=v_all)[
                    batch_idx, target_idx
                ]
                fxfycxcy_target = rearrange(fxfycxcy, "(b v) c -> b v c", v=v_all)[
                    batch_idx, target_idx
                ]
                c2w_target = rearrange(c2w_target, "b v c d -> (b v) c d")
                fxfycxcy_target = rearrange(fxfycxcy_target, "b v c -> (b v) c")
                vis_only_results = self.render_images_video(
                    scene_tokens, c2w_target, fxfycxcy_target, normalized=normalized
                )

        pred = render_results.rendered_images
        tgt = target.image
        if tgt.size(2) == 4:
            tgt = tgt[:, :, :3]

        pred_patch = rearrange(
            pred, "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)", ph=ph, pw=ph
        )
        tgt_patch = rearrange(
            tgt, "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)", ph=ph, pw=ph
        )

        per_sample_mse = []
        for i in range(pred_patch.shape[0]):
            use_mask = pmask_v[i] == 1
            if use_mask.any():
                pred_sel = pred_patch[i][use_mask]
                tgt_sel = tgt_patch[i][use_mask]
            else:
                pred_sel = pred_patch[i]
                tgt_sel = tgt_patch[i]
            per_sample_mse.append(F.mse_loss(pred_sel, tgt_sel))
        l2_loss = torch.stack(per_sample_mse).mean()
        psnr = -10.0 * torch.log10(l2_loss.clamp(min=1e-8))

        # Keep non-masked regions from GT for visualisation/lpips/perceptual if enabled.
        mask_f = pmask.unsqueeze(1).unsqueeze(-1).float().expand(-1, v_target, -1, -1)
        assembled_patch = tgt_patch.reshape(b, v_target, num_patches, -1).detach() * (
            1.0 - mask_f
        ) + pred_patch.reshape(b, v_target, num_patches, -1) * mask_f
        assembled = rearrange(
            assembled_patch,
            "b v (hh ww) (ph pw c) -> b v c (hh ph) (ww pw)",
            hh=hp,
            ww=wp,
            ph=ph,
            pw=ph,
            c=3,
        )

        lpips_loss = torch.tensor(0.0, device=device)
        if (
            self.config.training.lpips_loss_weight > 0.0
            and hasattr(self.loss_computer, "lpips_loss_module")
        ):
            bvt = assembled.shape[0] * assembled.shape[1]
            lpips_loss = self.loss_computer.lpips_loss_module(
                assembled.reshape(bvt, 3, hp * ph, wp * ph) * 2.0 - 1.0,
                tgt.reshape(bvt, 3, hp * ph, wp * ph) * 2.0 - 1.0,
            ).mean()

        perceptual_loss = torch.tensor(0.0, device=device)
        if (
            self.config.training.perceptual_loss_weight > 0.0
            and hasattr(self.loss_computer, "perceptual_loss_module")
        ):
            bvt = assembled.shape[0] * assembled.shape[1]
            perceptual_loss = self.loss_computer.perceptual_loss_module(
                assembled.reshape(bvt, 3, hp * ph, wp * ph),
                tgt.reshape(bvt, 3, hp * ph, wp * ph),
            )

        loss = (
            self.config.training.l2_loss_weight * l2_loss
            + self.config.training.lpips_loss_weight * lpips_loss
            + self.config.training.perceptual_loss_weight * perceptual_loss
        )
        loss_metrics = edict(
            loss=loss,
            l2_loss=l2_loss,
            psnr=psnr,
            lpips_loss=lpips_loss,
            perceptual_loss=perceptual_loss,
            norm_perceptual_loss=perceptual_loss / l2_loss.clamp(min=1e-8),
            norm_lpips_loss=lpips_loss / l2_loss.clamp(min=1e-8),
            # quick health signal: if this is ~0 for long time, multi-target outputs
            # are likely collapsing to near-identical predictions.
            pred_view_std=pred.std(dim=1).mean(),
        )

        result = edict(
            input=input,
            target=target,
            loss_metrics=loss_metrics,
            render=assembled,
            c2w=rearrange(c2w, "(b v) c d -> b v c d", b=b, v=v_all),
            fxfycxcy=fxfycxcy,
            input_idx=input_idx,
            target_idx=target_idx,
        )

        if create_visual and render_video:
            result.video_rendering = vis_only_results.rendered_images_video.detach()

        return result
