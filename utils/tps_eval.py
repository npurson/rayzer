"""
True Pose Similarity (TPS) Evaluation.

Adapted from XFactor (Mitchel et al., arXiv:2510.13063).
Uses VGGT as an oracle to evaluate whether rendered NVS images
preserve the 3D geometric structure of the original scene.

Usage:
    oracle = VggtOracle().to("cuda")
    tps = oracle.compute_tps(target_images, rendered_images)
"""

import math
from typing import Dict, Iterable, List, Literal, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# =============================================================================
# Image preprocessing
# =============================================================================

def resize_images_for_vggt(
    image_tensors: Tensor,
    mode: Literal["crop", "pad"] = "crop",
    target_size: int = 518,
    patch_size: int = 14,
) -> Tensor:
    """Resize image tensors for VGGT input, handling arbitrary leading dims."""
    if image_tensors.dim() < 3:
        raise ValueError("Input tensor must have at least 3 dimensions (C, H, W)")

    original_shape = image_tensors.shape
    *batch_dims, C, H, W = original_shape

    if mode == "pad":
        if W >= H:
            new_width = target_size
            new_height = round(H * (new_width / W) / patch_size) * patch_size
        else:
            new_height = target_size
            new_width = round(W * (new_height / H) / patch_size) * patch_size
    else:
        scale = target_size / min(H, W)
        new_height = int(math.ceil((H * scale) / patch_size) * patch_size)
        new_width = int(math.ceil((W * scale) / patch_size) * patch_size)

    flat = image_tensors.reshape(-1, C, H, W)
    resized = F.interpolate(flat, size=(new_height, new_width), mode="bicubic", align_corners=False)

    if mode == "crop":
        _, _, h, w = resized.shape
        sy = max(0, (h - target_size) // 2)
        sx = max(0, (w - target_size) // 2)
        resized = resized[:, :, sy:sy + target_size, sx:sx + target_size]
    elif mode == "pad":
        _, _, h, w = resized.shape
        hp, wp = target_size - h, target_size - w
        if hp > 0 or wp > 0:
            resized = F.pad(resized, (wp // 2, wp - wp // 2, hp // 2, hp - hp // 2), value=1.0)

    _, fc, fh, fw = resized.shape
    return resized.reshape(batch_dims + [fc, fh, fw])


# =============================================================================
# Camera metrics (from xfactor/vggt/metrics)
# =============================================================================

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def _acos_linear_extrapolation(
    x: Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> Tensor:
    lower, upper = bounds
    out = torch.empty_like(x)
    x_upper = x >= upper
    x_lower = x <= lower
    x_mid = (~x_upper) & (~x_lower)

    out[x_mid] = torch.acos(x[x_mid])
    out[x_upper] = (x[x_upper] - upper) * (-1.0 / math.sqrt(1.0 - upper * upper)) + math.acos(upper)
    out[x_lower] = (x[x_lower] - lower) * (-1.0 / math.sqrt(1.0 - lower * lower)) + math.acos(lower)
    return out


def _so3_rotation_angle(R: Tensor, cos_bound: float = 1e-4, eps: float = 1e-4) -> Tensor:
    N, d1, d2 = R.shape
    assert d1 == 3 and d2 == 3
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    phi_cos = (trace - 1.0) * 0.5
    bound = 1.0 - cos_bound
    return _acos_linear_extrapolation(phi_cos, (-bound, bound))


def _rotation_angle_deg(R_gt: Tensor, R_pred: Tensor, cos_bound: float = 1e-12) -> Tensor:
    R12 = torch.bmm(R_gt, R_pred.permute(0, 2, 1))
    angle_rad = _so3_rotation_angle(R12, cos_bound=cos_bound).clip(min=0)
    return angle_rad * 180.0 / torch.pi


def _translation_angle_deg(t_gt: Tensor, t_pred: Tensor, eps: float = 1e-12) -> Tensor:
    t_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + eps)
    t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + eps)
    loss = torch.clamp_min(1.0 - (t_gt * t_pred).sum(dim=1) ** 2, eps)
    err = torch.acos(torch.sqrt(1 - loss))
    err[torch.isnan(err) | torch.isinf(err)] = 1e6
    return err * 180.0 / torch.pi


def _batched_all_pairs(B: int, N: int):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1 = (i1_[None] + torch.arange(B)[:, None] * N).reshape(-1)
    i2 = (i2_[None] + torch.arange(B)[:, None] * N).reshape(-1)
    return i1, i2


@torch.no_grad()
def camera_to_rel_deg(pred_cameras: Tensor, gt_cameras: Tensor) -> Tuple[Tensor, Tensor]:
    """Compute pairwise relative rotation/translation angle errors in degrees."""
    pred_cameras = pred_cameras.to(torch.float64)
    gt_cameras = gt_cameras.to(torch.float64)
    device = pred_cameras.device
    B, S, _, _ = gt_cameras.shape

    pred_flat = pred_cameras.flatten(end_dim=1)
    gt_flat = gt_cameras.flatten(end_dim=1)

    i1, i2 = _batched_all_pairs(B, S)
    i1, i2 = i1.to(device), i2.to(device)

    rel_gt = gt_flat[i1].bmm(torch.linalg.inv(gt_flat[i2]))
    rel_pred = pred_flat[i1].bmm(torch.linalg.inv(pred_flat[i2]))

    r_deg = _rotation_angle_deg(rel_gt[:, :3, :3], rel_pred[:, :3, :3])
    t_deg = _translation_angle_deg(rel_gt[:, :3, 3], rel_pred[:, :3, 3])
    return r_deg, t_deg


def _normalized_histogram(errors: Tensor, max_threshold: int = 30) -> Tensor:
    hist = torch.histc(errors, bins=max_threshold + 1, min=0, max=max_threshold)
    return hist / float(len(errors))


def _scale_procrustes(target: Tensor, pred: Tensor) -> Tuple[Tensor, Tensor]:
    num = torch.einsum("...ij,...ij->...", target, pred)
    denom = torch.einsum("...ij,...ij->...", pred, pred)
    s = (num / torch.clamp(denom, min=1e-8)).detach()
    return target, s[..., None, None] * pred


@torch.no_grad()
def get_camera_metrics(
    *, target_T_cw: Tensor, pred_T_cw: Tensor
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """
    Full camera pose evaluation suite (Rac, Tac, AUC, trajectory error).

    Args:
        target_T_cw: [B, V, 4, 4] ground truth world-to-camera transforms
        pred_T_cw:   [B, V, 4, 4] predicted world-to-camera transforms

    Returns:
        (lower_the_better, higher_the_better) metric dicts, values shaped [B,]
    """
    target_T_cw = target_T_cw.clone().double()
    pred_T_cw = pred_T_cw.clone().double()
    B, V, _, _ = target_T_cw.shape

    # --- Lower the better ---
    lower = {}
    gt_pos = torch.linalg.inv(target_T_cw)[..., :3, 3].flatten(end_dim=-3)
    pred_pos = torch.linalg.inv(pred_T_cw)[..., :3, 3].flatten(end_dim=-3)
    _, aligned = _scale_procrustes(gt_pos, pred_pos)
    lower["traj_err_scale_invariant"] = torch.mean(torch.abs(gt_pos - aligned), dim=(-2, -1))
    lower["traj_err_unscaled"] = torch.mean(
        torch.abs(torch.linalg.inv(pred_T_cw)[..., :3, 3] - torch.linalg.inv(target_T_cw)[..., :3, 3]),
        dim=(-2, -1),
    )

    # --- Higher the better ---
    higher = {}
    r_deg, t_deg = camera_to_rel_deg(pred_T_cw, target_T_cw)
    r_deg = r_deg.reshape(B, V * (V - 1) // 2)
    t_deg = t_deg.reshape(B, V * (V - 1) // 2)

    if r_deg.numel() == 0:
        r_deg = torch.zeros(1, device=target_T_cw.device, dtype=target_T_cw.dtype)
        t_deg = torch.zeros(1, device=target_T_cw.device, dtype=target_T_cw.dtype)

    for th in [3, 5, 10, 15, 20, 30]:
        higher[f"Rac_{th}"] = (r_deg < th).double().mean(dim=-1)
        higher[f"Tac_{th}"] = (t_deg < th).double().mean(dim=-1)

    rot_hist = torch.stack([_normalized_histogram(r_deg[b]) for b in range(B)])
    trans_hist = torch.stack([_normalized_histogram(t_deg[b]) for b in range(B)])
    se3_hist = torch.stack([
        _normalized_histogram(torch.stack([r_deg[b], t_deg[b]], dim=-1).max(dim=-1).values)
        for b in range(B)
    ])

    for ath in [3, 5, 10, 15, 20, 30]:
        higher[f"Rot_Auc_{ath}"] = torch.cumsum(rot_hist[:, :ath], dim=-1).mean(dim=-1)
        higher[f"Trans_Auc_{ath}"] = torch.cumsum(trans_hist[:, :ath], dim=-1).mean(dim=-1)
        higher[f"SE3_Auc_{ath}"] = torch.cumsum(se3_hist[:, :ath], dim=-1).mean(dim=-1)

    return lower, higher


# =============================================================================
# VGGT Oracle
# =============================================================================

class VggtOracle(nn.Module):
    """
    Frozen VGGT model used as an oracle for TPS evaluation.

    Loads VGGT weights (from HuggingFace Hub by default), runs pose-only
    inference, and compares camera trajectories between two sets of images.
    """

    IMG_SIZE: int = 518

    def __init__(self, checkpoint_path: str = "facebook/VGGT-1B"):
        super().__init__()
        print("Initializing VGGT oracle...", flush=True)
        self.vggt = VGGT(
            img_size=self.IMG_SIZE,
            enable_point=False,
            enable_depth=False,
            enable_track=False,
        )
        print(f"Loading VGGT weights from {checkpoint_path}...", flush=True)
        if checkpoint_path == "facebook/VGGT-1B":
            _state = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
                map_location="cpu",
            )
        else:
            _state = torch.load(checkpoint_path, map_location="cpu")
        self.vggt.load_state_dict(_state, strict=False)
        self.vggt.eval()
        for p in self.vggt.parameters():
            p.requires_grad = False

    @property
    def device(self):
        return next(self.vggt.parameters()).device

    # -----------------------------------------------------------------
    # Core forward: images -> T_cw
    # -----------------------------------------------------------------
    @torch.no_grad()
    def predict_poses(
        self,
        images: Tensor,
        *,
        batch_chunk_size: int | None = None,
        preprocess: bool = True,
        ref_frame_id: int = 0,
    ) -> Tensor:
        """
        Predict camera poses for a batch of multi-view image sequences.

        Args:
            images: [B, V, 3, H, W] or [V, 3, H, W], values in [0, 1]
            batch_chunk_size: split batch to save memory
            preprocess: whether to resize/crop to 518x518
            ref_frame_id: which frame is the reference (identity pose)

        Returns:
            T_cw: [B, V, 4, 4] world-to-camera transforms
        """
        if images.ndim == 4:
            images = images.unsqueeze(0)
        images = images.to(device=self.device, dtype=torch.float32).clamp(0, 1)

        if preprocess:
            images = resize_images_for_vggt(images, mode="crop")

        B, S = images.shape[:2]
        if ref_frame_id < 0:
            ref_frame_id = S + ref_frame_id

        # VGGT performs best when the reference frame is at index 0.
        # Reorder so ref_frame sits at position 0, then undo afterwards.
        im1 = images[:, :ref_frame_id]
        im_ref = images[:, ref_frame_id:ref_frame_id + 1]
        im2 = images[:, ref_frame_id + 1:]
        images_reordered = torch.cat([im_ref, im1, im2], dim=1)

        if batch_chunk_size is None:
            batch_chunk_size = B
        chunks = torch.chunk(images_reordered, math.ceil(B / batch_chunk_size))

        T_cw_list = []
        for chunk in chunks:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                preds = self.vggt(chunk)
            pose_enc = preds["pose_enc"]
            T_cw, _K = pose_encoding_to_extri_intri(pose_enc, chunk.shape[-2:])
            T_cw = torch.cat([T_cw, torch.zeros_like(T_cw[..., :1, :])], dim=-2)
            T_cw[..., -1, -1] = 1.0
            T_cw_list.append(T_cw)

        T_cw = torch.cat(T_cw_list, dim=0)

        # Undo the reordering
        v2 = T_cw[:, :1]          # was ref
        v1 = T_cw[:, 1:ref_frame_id + 1]
        v3 = T_cw[:, ref_frame_id + 1:]
        T_cw = torch.cat([v1, v2, v3], dim=1)

        return T_cw  # [B, V, 4, 4]

    # -----------------------------------------------------------------
    # TPS computation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def compute_tps(
        self,
        target_images: Tensor,
        rendered_images: Tensor,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compute True Pose Similarity between target (GT) and rendered images.

        Both inputs should be multi-view sequences of the same scene,
        shaped [B, V, 3, H, W] or [V, 3, H, W], values in [0, 1].

        Returns:
            dict of metric_name -> Tensor[B,] (lower_the_better metrics).
            Key metrics: traj_err_scale_invariant, traj_err_unscaled.
        """
        target_T = self.predict_poses(target_images, **kwargs)
        render_T = self.predict_poses(rendered_images, **kwargs)
        lower, higher = get_camera_metrics(target_T_cw=target_T, pred_T_cw=render_T)
        return {**lower, **higher}

    # -----------------------------------------------------------------
    # Convenience: evaluate a batch from rayzer inference result
    # -----------------------------------------------------------------
    @torch.no_grad()
    def evaluate_batch(
        self,
        gt_images: Tensor,
        rendered_images: Tensor,
        context_images: Tensor | None = None,
    ) -> Dict[str, float]:
        """
        Evaluate TPS for a single inference batch.

        Args:
            gt_images: [V_target, 3, H, W] GT target views
            rendered_images: [V_target, 3, H, W] rendered target views
            context_images: [V_ctx, 3, H, W] optional context views to prepend,
                forming a longer sequence for more robust pose estimation.

        Returns:
            dict of scalar metrics
        """
        if context_images is not None:
            gt_seq = torch.cat([context_images, gt_images], dim=0).unsqueeze(0)
            render_seq = torch.cat([context_images, rendered_images], dim=0).unsqueeze(0)
        else:
            gt_seq = gt_images.unsqueeze(0)
            render_seq = rendered_images.unsqueeze(0)

        metrics = self.compute_tps(gt_seq, render_seq)
        return {k: v.item() for k, v in metrics.items()}
