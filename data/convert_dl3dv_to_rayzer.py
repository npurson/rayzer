"""
将 DL3DV-10K 的 nerfstudio transforms.json 格式转换为 RayZer 的 opencv_cameras.json 格式。

使用方法：
    python convert_dl3dv_to_rayzer.py \
        --src /horizon-bucket/robot_lab/datasets/DL3DV-10K/480P \
        --dst /horizon-bucket/robot_lab/users/haoyi.jiang/data/dl3dv_rayzer_10k \
        --workers 16

格式差异：
- DL3DV transforms.json: nerfstudio/OpenGL convention, c2w (transform_matrix), global intrinsics at original resolution
- RayZer opencv_cameras.json: OpenCV convention, w2c, per-frame intrinsics at actual image resolution
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image
import traceback


# OpenGL -> OpenCV 坐标变换矩阵 (翻转 y 和 z 轴)
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def convert_scene(scene_info, dst_root):
    """转换单个场景"""
    subset, scene_hash, scene_dir = scene_info
    
    try:
        transforms_path = os.path.join(scene_dir, 'transforms.json')
        images_dir = os.path.join(scene_dir, 'images_8')
        
        if not os.path.exists(transforms_path):
            return f"SKIP (no transforms.json): {scene_hash}"
        if not os.path.isdir(images_dir):
            return f"SKIP (no images_8): {scene_hash}"
        
        # 读取 transforms.json
        with open(transforms_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if len(frames) == 0:
            return f"SKIP (no frames): {scene_hash}"
        
        # 获取实际图片尺寸
        first_image_name = os.path.basename(frames[0]['file_path'])
        first_image_path = os.path.join(images_dir, first_image_name)
        if not os.path.exists(first_image_path):
            return f"SKIP (image not found): {scene_hash}, {first_image_path}"
        
        img = Image.open(first_image_path)
        actual_w, actual_h = img.size
        img.close()
        
        # 计算内参缩放因子
        orig_w = data['w']
        orig_h = data['h']
        scale_x = actual_w / orig_w
        scale_y = actual_h / orig_h
        
        # 全局内参（缩放到实际图片分辨率）
        global_fx = data['fl_x'] * scale_x
        global_fy = data['fl_y'] * scale_y
        global_cx = data['cx'] * scale_x
        global_cy = data['cy'] * scale_y
        
        # 转换每一帧
        converted_frames = []
        for frame in frames:
            # nerfstudio c2w (OpenGL convention) -> OpenCV c2w -> w2c
            c2w_opengl = np.array(frame['transform_matrix'])
            
            # 确保是 4x4
            if c2w_opengl.shape == (3, 4):
                c2w_4x4 = np.eye(4)
                c2w_4x4[:3, :] = c2w_opengl
                c2w_opengl = c2w_4x4
            
            c2w_opencv = c2w_opengl @ OPENGL_TO_OPENCV
            w2c_opencv = np.linalg.inv(c2w_opencv)
            
            # 获取 per-frame intrinsics (如果有的话) 或使用全局值
            fx = frame.get('fl_x', data['fl_x']) * scale_x
            fy = frame.get('fl_y', data['fl_y']) * scale_y
            cx = frame.get('cx', data['cx']) * scale_x
            cy = frame.get('cy', data['cy']) * scale_y
            
            # 图片文件名（相对路径，使用 images_8 目录）
            image_basename = os.path.basename(frame['file_path'])
            file_path = f"images_8/{image_basename}"
            
            converted_frames.append({
                "file_path": file_path,
                "w2c": w2c_opencv.tolist(),
                "h": actual_h,
                "w": actual_w,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            })
        
        # 构建输出 JSON
        output = {
            "scene_name": scene_hash,
            "frames": converted_frames,
        }
        
        # 创建输出目录
        out_scene_dir = os.path.join(dst_root, scene_hash)
        os.makedirs(out_scene_dir, exist_ok=True)
        
        # 写入 opencv_cameras.json
        out_json_path = os.path.join(out_scene_dir, 'opencv_cameras.json')
        with open(out_json_path, 'w') as f:
            json.dump(output, f)
        
        # 创建指向原始 images_8 的符号链接
        link_path = os.path.join(out_scene_dir, 'images_8')
        if not os.path.exists(link_path):
            os.symlink(images_dir, link_path)
        
        return f"OK: {scene_hash} ({len(converted_frames)} frames, {actual_w}x{actual_h})"
    
    except Exception as e:
        return f"ERROR: {scene_hash}: {traceback.format_exc()}"


def collect_all_scenes(src_root):
    """收集所有场景路径"""
    scenes = []
    subsets = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']
    
    for subset in subsets:
        subset_dir = os.path.join(src_root, subset)
        if not os.path.isdir(subset_dir):
            print(f"Warning: {subset_dir} not found, skipping")
            continue
        
        for scene_hash in sorted(os.listdir(subset_dir)):
            scene_dir = os.path.join(subset_dir, scene_hash)
            if os.path.isdir(scene_dir):
                scenes.append((subset, scene_hash, scene_dir))
    
    return scenes


def generate_train_txt(dst_root, output_txt_path):
    """生成 train.txt 文件"""
    json_paths = []
    for scene_hash in sorted(os.listdir(dst_root)):
        json_path = os.path.join(dst_root, scene_hash, 'opencv_cameras.json')
        if os.path.exists(json_path):
            json_paths.append(json_path)
    
    with open(output_txt_path, 'w') as f:
        for path in json_paths:
            f.write(path + '\n')
    
    return len(json_paths)


def main():
    parser = argparse.ArgumentParser(description='Convert DL3DV-10K to RayZer format')
    parser.add_argument('--src', type=str, required=True,
                        help='DL3DV-10K root (e.g., /horizon-bucket/.../DL3DV-10K/480P)')
    parser.add_argument('--dst', type=str, required=True,
                        help='Output directory for RayZer format data')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel workers')
    parser.add_argument('--train_txt', type=str, default=None,
                        help='Output train.txt path (default: <dst>/dl3dv_10k_train.txt)')
    args = parser.parse_args()
    
    os.makedirs(args.dst, exist_ok=True)
    
    # 收集所有场景
    print(f"Collecting scenes from {args.src} ...")
    scenes = collect_all_scenes(args.src)
    print(f"Found {len(scenes)} scenes")
    
    # 并行转换
    print(f"Converting with {args.workers} workers ...")
    convert_fn = partial(convert_scene, dst_root=args.dst)
    
    ok_count = 0
    skip_count = 0
    error_count = 0
    
    with Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(convert_fn, scenes, chunksize=32)):
            if result.startswith("OK"):
                ok_count += 1
            elif result.startswith("SKIP"):
                skip_count += 1
                print(result)
            else:
                error_count += 1
                print(result)
            
            if (i + 1) % 500 == 0:
                print(f"Progress: {i+1}/{len(scenes)} (OK={ok_count}, SKIP={skip_count}, ERROR={error_count})")
    
    print(f"\nDone! OK={ok_count}, SKIP={skip_count}, ERROR={error_count}")
    
    # 生成 train.txt
    train_txt_path = args.train_txt or os.path.join(args.dst, 'dl3dv_10k_train.txt')
    n = generate_train_txt(args.dst, train_txt_path)
    print(f"Generated {train_txt_path} with {n} scenes")


if __name__ == '__main__':
    main()
