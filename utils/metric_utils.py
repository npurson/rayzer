# Copyright (c) 2025 Hanwen Jiang. Created for the RayZer project.

import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
from skimage.metrics import structural_similarity
import functools
import os
from PIL import Image
from utils import data_utils
import numpy as np
from easydict import EasyDict as edict
import json
from rich import print

from utils.camera_eval import evaluate_camera_pose_metrics

import warnings
# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")

@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()



@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    """
    Compute Structural Similarity Index between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        SSIM values for each image in the batch (higher is better)
    """
    ssim_values= []
    
    for gt, pred in zip(ground_truth, predicted):
        # Move to CPU and convert to numpy
        gt_np = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        
        # Calculate SSIM
        ssim = structural_similarity(
            gt_np,
            pred_np,
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        ssim_values.append(ssim)
    
    # Convert back to tensor on the same device as input
    return torch.tensor(ssim_values, dtype=predicted.dtype, device=predicted.device)



@torch.no_grad()
def export_results(
    result: edict,
    out_dir: str, 
    compute_metrics: bool = False
):
    """
    Save results including images and optional metrics and videos.
    
    Args:
        result: EasyDict containing input, target, and rendered images, and optionally video frames
        out_dir: Directory to save the evaluation results
        compute_metrics: Whether to compute and save metrics
    """
    os.makedirs(out_dir, exist_ok=True)
    
    input_data, target_data = result.input, result.target
    
    for batch_idx in range(input_data.image.size(0)):
        uid = input_data.index[batch_idx, 0, -1].item()
        scene_name = input_data.scene_name[batch_idx]
        sample_dir = os.path.join(out_dir, f"{uid:06d}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get target view indices
        target_indices = target_data.index[batch_idx, :, 0].cpu().numpy()
        
        # Save images
        _save_images(result, batch_idx, sample_dir)
        
        # Compute and save metrics if requested
        if compute_metrics:
            _save_metrics(
                target_data.image[batch_idx],
                result.render[batch_idx],
                target_indices,
                sample_dir,
                scene_name
            )

            # Compute and save pose metrics if GT c2w is available
            if hasattr(result, 'gt_c2w') and result.gt_c2w is not None and hasattr(result, 'c2w'):
                _save_pose_metrics(
                    result.gt_c2w[batch_idx],   # [v_all, 4, 4]
                    result.c2w[batch_idx],       # [v_all, 4, 4]
                    sample_dir,
                    scene_name
                )
        
        # Save video if available
        if hasattr(result, "video_rendering"):
            _save_video(result.video_rendering[batch_idx], sample_dir)  # [v, c, h, w]


def visualize_intermediate_results(out_dir, result, save_all=False):
    os.makedirs(out_dir, exist_ok=True)

    input, target = result.input, result.target

    if result.render is not None:
        target_image = target.image
        rendered_image = result.render
        b, v, _, h, w = rendered_image.size()
        rendered_image = rendered_image.reshape(b * v, -1, h, w)
        target_image = target_image.reshape(b * v, -1, h, w)
        visualized_image = torch.cat((target_image, rendered_image), dim=3).detach().cpu()
        visualized_image = rearrange(visualized_image, "(b v) c h (m w) -> (b h) (v m w) c", v=v, m=2)
        visualized_image = (visualized_image.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        
        uids = [target.index[b, 0, -1].item() for b in range(target.index.size(0))]

        uid_based_filename = f"{uids[0]:08}_{uids[-1]:08}"
        Image.fromarray(visualized_image).save(
            os.path.join(out_dir, f"supervision_{uid_based_filename}.jpg")
        )
        with open(os.path.join(out_dir, f"uids.txt"), "w") as f:
            uids = "_".join([f"{uid:08}" for uid in uids])
            f.write(uids)

    # Save the input image grid
    input_uids = [input.index[b, 0, -1].item() for b in range(input.index.size(0))]
    input_uid_based_filename = f"{input_uids[0]:08}_{input_uids[-1]:08}"
    b, v, c, h, w = input.image.size()
    input_images = input.image.reshape(b * v, c, h, w).detach().cpu()
    input_grid = rearrange(input_images, "(b v) c h w -> (b h) (v w) c", v=v)
    input_grid = (input_grid.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_grid).save(
        os.path.join(out_dir, f"input_{input_uid_based_filename}.jpg")
    )

    # Save input, rendered interpolated video
    for b in range(input.image.size(0)):
        uid = input.index[b, 0, -1].item()
        
        # vis input
        vis_image = rearrange(input.image[b], "v c h w -> h (v w) c")
        vis_image = (vis_image.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
        Image.fromarray(vis_image).save(os.path.join(out_dir, f"input_{uid}.png"))

        if ('video_rendering' in result.keys()) and torch.is_tensor(result.video_rendering):
            video_rendering = result.video_rendering[b]   # [v, c, h, w]
            _save_video(video_rendering, os.path.join(out_dir, f"render_video_{uid}.mp4"))

        if not save_all:
            break


def _save_images(result, batch_idx, out_dir):
    """Save visualization images."""
    # Save input image
    input_img = result.input.image[batch_idx]
    input_img = rearrange(input_img, "v c h w -> h (v w) c")
    input_img = (input_img.cpu().numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(input_img).save(os.path.join(out_dir, "input.png"))

    # Save GT vs prediction side-by-side
    comparison = torch.cat(
        (result.target.image[batch_idx], result.render[batch_idx]), 
        dim=2
    ).detach().cpu()
    comparison = rearrange(comparison, "v c h w -> h (v w) c")
    comparison = (comparison.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(comparison).save(os.path.join(out_dir, "gt_vs_pred.png"))
    

def _save_metrics(target, prediction, view_indices, out_dir, scene_name):
    target = target.to(torch.float32)
    prediction = prediction.to(torch.float32)
    
    psnr_values = compute_psnr(target, prediction)
    lpips_values = compute_lpips(target, prediction)
    ssim_values = compute_ssim(target, prediction)

    metrics = {
        "summary": {
            "scene_name": scene_name,
            "psnr": float(psnr_values.mean()),
            "lpips": float(lpips_values.mean()),
            "ssim": float(ssim_values.mean())
        },
        "per_view": []
    }
    
    for i, view_idx in enumerate(view_indices):
        metrics["per_view"].append({
            "view": int(view_idx), "psnr": float(psnr_values[i]), "lpips": float(lpips_values[i]), "ssim": float(ssim_values[i])
        })
    
    # Save metrics to a single JSON file
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


def _save_pose_metrics(gt_c2w, pred_c2w, out_dir, scene_name):
    """
    Evaluate and save camera pose metrics.
    
    Args:
        gt_c2w: [V, 4, 4] ground truth c2w matrices
        pred_c2w: [V, 4, 4] predicted c2w matrices
        out_dir: directory to save metrics
        scene_name: scene identifier
    """
    gt_c2w = gt_c2w.to(torch.float32)
    pred_c2w = pred_c2w.to(torch.float32)
    
    pose_metrics = evaluate_camera_pose_metrics(gt_c2w, pred_c2w, thresholds=[5, 15, 30])
    
    result = {
        "scene_name": scene_name,
        **{k: float(v) for k, v in pose_metrics.items()}
    }
    
    with open(os.path.join(out_dir, "pose_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)


def _save_video(frames, output_path):
    """
    Save video from rendered frames.
    Input frames should be in [v, c, h, w] format.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    frames = np.ascontiguousarray(np.array(frames.to(torch.float32).detach().cpu()))
    frames = rearrange(frames, "v c h w -> v h w c")
    
    # Handle NaN and Inf values (can occur early in training)
    if not np.isfinite(frames).all():
        print(f"Warning: Video frames contain NaN/Inf values, replacing with 0")
        frames = np.nan_to_num(frames, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Clip to valid range
    frames = np.clip(frames, 0.0, 1.0)
    
    data_utils.create_video_from_frames(
        frames, 
        output_path, 
        framerate=30
    )


def summarize_evaluation(evaluation_folder):
    # Find and sort all valid subfolders
    subfolders = sorted(
        [
            os.path.join(evaluation_folder, dirname)
            for dirname in os.listdir(evaluation_folder)
            if os.path.isdir(os.path.join(evaluation_folder, dirname))
        ],
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else os.path.basename(x)
    )

    metrics = {}
    pose_metrics = {}
    valid_subfolders = []
    pose_valid_subfolders = []
    
    for subfolder in subfolders:
        # Rendering metrics
        json_path = os.path.join(subfolder, "metrics.json")
        if not os.path.exists(json_path):
            print(f"!!! Metrics file not found in {subfolder}, skipping...")
            continue
            
        valid_subfolders.append(subfolder)
        
        with open(json_path, "r") as f:
            try:
                data = json.load(f)
                for metric_name, metric_value in data["summary"].items():
                    if metric_name == "scene_name":
                        continue
                    metrics.setdefault(metric_name, []).append(metric_value)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading metrics from {json_path}: {e}")

        # Pose metrics
        pose_json_path = os.path.join(subfolder, "pose_metrics.json")
        if os.path.exists(pose_json_path):
            pose_valid_subfolders.append(subfolder)
            with open(pose_json_path, "r") as f:
                try:
                    pose_data = json.load(f)
                    for metric_name, metric_value in pose_data.items():
                        if metric_name == "scene_name":
                            continue
                        pose_metrics.setdefault(metric_name, []).append(metric_value)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error reading pose metrics from {pose_json_path}: {e}")

    if not valid_subfolders:
        print(f"No valid metrics files found in {evaluation_folder}")
        return

    # === Rendering metrics summary ===
    csv_file = os.path.join(evaluation_folder, "summary.csv")
    with open(csv_file, "w") as f:
        header = ["Index"] + list(metrics.keys())
        f.write(",".join(header) + "\n")
        
        for i, subfolder in enumerate(valid_subfolders):
            basename = os.path.basename(subfolder)
            values = [str(metric_values[i]) for metric_values in metrics.values()]
            f.write(f"{basename},{','.join(values)}\n")
        
        f.write("\n")
        
        averages = [str(sum(values) / len(values)) for values in metrics.values()]
        f.write(f"average,{','.join(averages)}\n")
    
    print(f"[Rendering] Summary written to {csv_file}")
    print(f"[Rendering] Average: {','.join(averages)}")

    # === Pose metrics summary ===
    if pose_metrics:
        pose_csv_file = os.path.join(evaluation_folder, "pose_summary.csv")
        with open(pose_csv_file, "w") as f:
            header = ["Index"] + list(pose_metrics.keys())
            f.write(",".join(header) + "\n")
            
            for i, subfolder in enumerate(pose_valid_subfolders):
                basename = os.path.basename(subfolder)
                values = [str(metric_values[i]) for metric_values in pose_metrics.values()]
                f.write(f"{basename},{','.join(values)}\n")
            
            f.write("\n")
            
            pose_averages = [str(sum(values) / len(values)) for values in pose_metrics.values()]
            f.write(f"average,{','.join(pose_averages)}\n")
        
        print(f"[Pose] Summary written to {pose_csv_file}")
        pose_avg_str = ", ".join(f"{k}: {sum(v)/len(v):.4f}" for k, v in pose_metrics.items())
        print(f"[Pose] Average: {pose_avg_str}")

    # Export combined average metrics
    with open(os.path.join(evaluation_folder, "average_metrics.txt"), "w") as f:
        render_header = list(metrics.keys())
        f.write(f"[Rendering] {', '.join(render_header)}\n")
        f.write(f"[Rendering] Average: {', '.join(averages)}\n")
        if pose_metrics:
            pose_header = list(pose_metrics.keys())
            f.write(f"\n[Pose] {', '.join(pose_header)}\n")
            f.write(f"[Pose] Average: {', '.join(pose_averages)}\n")
