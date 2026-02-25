"""
Camera Pose Evaluation Utilities

Adapted from:
https://github.com/facebookresearch/vggt/blob/evaluation/evaluation/test_co3d.py#L163
"""

import torch
import numpy as np


def closed_form_inverse_se3(se3):
    R = se3[:, :3, :3]
    T = se3[:, :3, 3:]

    R_T = R.transpose(1, 2)
    t_new = -torch.bmm(R_T, T)

    inv = (
        torch.eye(4, device=se3.device, dtype=se3.dtype)
        .unsqueeze(0)
        .repeat(se3.size(0), 1, 1)
    )
    inv[:, :3, :3] = R_T
    inv[:, :3, 3:] = t_new

    return inv


def build_pair_index(N):
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    return i1_, i2_


def rotation_angle_deg(R_gt, R_pred, eps=1e-15):
    rel_R = torch.bmm(R_gt.transpose(1, 2), R_pred)
    trace = rel_R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
    angle = torch.arccos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    return angle * 180.0 / np.pi


def translation_angle_deg(t_gt, t_pred, eps=1e-15):
    t_gt = t_gt / (t_gt.norm(dim=1, keepdim=True) + eps)
    t_pred = t_pred / (t_pred.norm(dim=1, keepdim=True) + eps)
    cos = torch.clamp((t_gt * t_pred).sum(dim=1), -1 + eps, 1 - eps)
    angle = torch.acos(cos)
    return torch.min(angle, np.pi - angle) * 180.0 / np.pi


def calculate_rpa(errors, thresholds):
    sorted_errs = np.sort(errors)
    rpas = {}
    for t in thresholds:
        covered = (sorted_errs <= t).sum()
        rpa = (covered / len(sorted_errs)) if len(sorted_errs) > 0 else 0.0
        rpas[f"RPA@{t}"] = rpa
    return rpas


def evaluate_camera_pose_metrics(gt_c2w, pred_c2w, thresholds=[5, 15, 30]):
    """
    Evaluate relative camera pose accuracy between GT and predicted c2w matrices.

    Args:
        gt_c2w: [V, 4, 4] ground truth camera-to-world matrices
        pred_c2w: [V, 4, 4] predicted camera-to-world matrices
        thresholds: list of angle thresholds in degrees for RPA computation

    Returns:
        dict with RPA@threshold values, mean rotation error, and mean translation error
    """
    device = gt_c2w.device

    V = gt_c2w.shape[0]
    if V < 2:
        return {
            "RPA@5": 0.0,
            "RPA@15": 0.0,
            "RPA@30": 0.0,
            "rot_err_mean": 0.0,
            "trans_err_mean": 0.0,
        }

    i1, i2 = build_pair_index(V)
    i1, i2 = i1.to(device), i2.to(device)

    gt_w2c = torch.inverse(gt_c2w)
    pred_w2c = torch.inverse(pred_c2w)

    rel_gt = torch.bmm(gt_w2c[i1], closed_form_inverse_se3(gt_w2c[i2]))
    rel_pred = torch.bmm(pred_w2c[i1], closed_form_inverse_se3(pred_w2c[i2]))

    r_err = rotation_angle_deg(rel_gt[:, :3, :3], rel_pred[:, :3, :3]).cpu().numpy()
    t_err = translation_angle_deg(rel_gt[:, :3, 3], rel_pred[:, :3, 3]).cpu().numpy()

    max_err = np.maximum(r_err, t_err)

    # Compute RPAs
    rpas = calculate_rpa(max_err, thresholds=thresholds)

    # Also return mean errors for detailed analysis
    rpas["rot_err_mean"] = float(np.mean(r_err))
    rpas["trans_err_mean"] = float(np.mean(t_err))

    return rpas
