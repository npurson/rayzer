# Copyright (c) 2025 Haian Jin and Hanwen Jiang. Created for the LVSM project (ICLR 2025) and RayZer (ICCV 2025).

import random
import os
import numpy as np
import PIL
import torch
from torch.utils.data import Dataset
import json
import torch.nn.functional as F


class Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.current_iteration = 0

        try:
            with open(self.config.training.dataset_path, "r") as f:
                self.all_scene_paths = f.read().splitlines()
            self.all_scene_paths = [
                path for path in self.all_scene_paths if path.strip()
            ]

        except Exception as e:
            print(
                f"Error reading dataset paths from '{self.config.training.dataset_path}'"
            )
            raise e

        self.inference = self.config.inference.get("if_inference", False)
        # Load file that specifies the input and target view indices to use for inference
        if self.inference:
            self.view_idx_list = dict()
            if self.config.inference.get("view_idx_file_path", None) is not None:
                if os.path.exists(self.config.inference.view_idx_file_path):
                    with open(self.config.inference.view_idx_file_path, "r") as f:
                        self.view_idx_list = json.load(f)
                        # filter out None values, i.e. scenes that don't have specified input and targetviews
                        self.view_idx_list_filtered = [
                            k for k, v in self.view_idx_list.items() if v is not None
                        ]
                    filtered_scene_paths = []
                    for scene in self.all_scene_paths:
                        # file_name = scene.split("/")[-1]
                        # scene_name = file_name.split(".")[0]
                        scene_name = scene.split("/")[-2]
                        if scene_name in self.view_idx_list_filtered:
                            filtered_scene_paths.append(scene)

                    self.all_scene_paths = filtered_scene_paths

    def __len__(self):
        return len(self.all_scene_paths)

    def update_iteration(self, iteration):
        self.current_iteration = iteration
        curriculum_max_iter = self.config.training.view_selector.get(
            "curriculum_iter", 30000
        )
        progress = min(iteration / curriculum_max_iter, 1.0)

        min_frame_dist = self.config.training.view_selector.get("min_frame_dist", 25)
        max_frame_dist = self.config.training.view_selector.get("max_frame_dist", 100)
        min_frame_dist_start, max_frame_dist_start = (
            self.config.training.view_selector.get(
                "curriculum_start_min_frame_dist", 48
            ),
            self.config.training.view_selector.get(
                "curriculum_start_max_frame_dist", 64
            ),
        )
        cur_min_frame_dist = int(
            min_frame_dist_start + (min_frame_dist - min_frame_dist_start) * progress
        )
        cur_max_frame_dist = int(
            max_frame_dist_start + (max_frame_dist - max_frame_dist_start) * progress
        )

        if self.config.training.view_selector.get("use_curriculum", False):
            if (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
            ):
                print(
                    "Current curriculum progress: {}, min_dist: {}, max_dist: {}".format(
                        progress, cur_min_frame_dist, cur_max_frame_dist
                    )
                )

    def preprocess_frames(self, frames_chosen, image_paths_chosen):
        target_size = self.config.model.image_tokenizer.image_size
        patch_size = self.config.model.image_tokenizer.patch_size
        square_crop = self.config.training.get("square_crop", False)

        images = []
        intrinsics = []
        for cur_frame, cur_image_path in zip(frames_chosen, image_paths_chosen):
            image = PIL.Image.open(cur_image_path)
            original_image_w, original_image_h = image.size

            if square_crop:
                # 确保短边 >= target_size，长边按比例缩放，然后中心裁剪到 target_size x target_size
                if original_image_w >= original_image_h:
                    # 横向图：以高度为基准
                    resize_h = target_size
                    resize_w = int(target_size / original_image_h * original_image_w)
                    resize_w = int(round(resize_w / patch_size) * patch_size)
                else:
                    # 纵向图：以宽度为基准，确保宽度 >= target_size
                    resize_w = int(round(target_size / patch_size) * patch_size)
                    resize_h = int(resize_w / original_image_w * original_image_h)
                    resize_h = int(round(resize_h / patch_size) * patch_size)
            else:
                resize_h = target_size
                resize_w = int(target_size / original_image_h * original_image_w)
                resize_w = int(round(resize_w / patch_size) * patch_size)

            image = image.resize((resize_w, resize_h), resample=PIL.Image.LANCZOS)
            if square_crop:
                crop_size = target_size
                start_h = (resize_h - crop_size) // 2
                start_w = (resize_w - crop_size) // 2
                image = image.crop(
                    (start_w, start_h, start_w + crop_size, start_h + crop_size)
                )

            image = np.array(image) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            fxfycxcy = np.array(
                [cur_frame["fx"], cur_frame["fy"], cur_frame["cx"], cur_frame["cy"]]
            )
            resize_ratio_x = resize_w / original_image_w
            resize_ratio_y = resize_h / original_image_h
            fxfycxcy *= (resize_ratio_x, resize_ratio_y, resize_ratio_x, resize_ratio_y)
            if square_crop:
                fxfycxcy[2] -= start_w
                fxfycxcy[3] -= start_h
            fxfycxcy = torch.from_numpy(fxfycxcy).float()
            images.append(image)
            intrinsics.append(fxfycxcy)

        images = torch.stack(images, dim=0)
        intrinsics = torch.stack(intrinsics, dim=0)
        w2cs = np.stack([np.array(frame["w2c"]) for frame in frames_chosen])
        c2ws = np.linalg.inv(w2cs)  # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        return images, intrinsics, c2ws

    def preprocess_poses(self, in_c2ws: torch.Tensor, scene_scale_factor=1.35):
        """
        Preprocess the poses to:
        1. translate and rotate the scene to align the average camera direction and position
        2. rescale the whole scene to a fixed scale
        """

        # Translation and Rotation
        # align coordinate system (OpenCV coordinate) to the mean camera
        # center is the average of all camera centers
        # average direction vectors are computed from all camera direction vectors (average down and forward)
        center = in_c2ws[:, :3, 3].mean(0)
        avg_forward = F.normalize(
            in_c2ws[:, :3, 2].mean(0), dim=-1
        )  # average forward direction (z of opencv camera)
        avg_down = in_c2ws[:, :3, 1].mean(
            0
        )  # average down direction (y of opencv camera)
        avg_right = F.normalize(
            torch.cross(avg_down, avg_forward, dim=-1), dim=-1
        )  # (x of opencv camera)
        avg_down = F.normalize(
            torch.cross(avg_forward, avg_right, dim=-1), dim=-1
        )  # (y of opencv camera)

        avg_pose = torch.eye(4, device=in_c2ws.device)  # average c2w matrix
        avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
        avg_pose[:3, 3] = center
        avg_pose = torch.linalg.inv(avg_pose)  # average w2c matrix
        in_c2ws = avg_pose @ in_c2ws

        # Rescale the whole scene to a fixed scale
        scene_scale = torch.max(torch.abs(in_c2ws[:, :3, 3]))
        scene_scale = scene_scale_factor * scene_scale

        in_c2ws[:, :3, 3] /= scene_scale

        return in_c2ws

    def view_selector(self, frames, current_iteration):
        if len(frames) < self.config.training.num_views:
            return None
        # sample view candidates
        view_selector_config = self.config.training.view_selector
        num_views = self.config.training.num_views

        use_curriculum = view_selector_config.get("use_curriculum", False)
        if view_selector_config.type == "two_frame" and (not use_curriculum):
            min_frame_dist = view_selector_config.get("min_frame_dist", 25)
            max_frame_dist = min(
                len(frames) - 1, view_selector_config.get("max_frame_dist", 100)
            )
            if max_frame_dist <= min_frame_dist:
                return None
            return two_frame_selector(frames, num_views, min_frame_dist, max_frame_dist)
        elif view_selector_config.type == "two_frame" and use_curriculum:
            min_frame_dist = view_selector_config.get("min_frame_dist", 25)
            max_frame_dist = min(
                len(frames) - 1, view_selector_config.get("max_frame_dist", 100)
            )
            curriculum_max_iter = view_selector_config.get("curriculum_iter", 30000)
            progress = min(current_iteration / curriculum_max_iter, 1.0)
            min_frame_dist_start = view_selector_config.get(
                "curriculum_start_min_frame_dist", 48
            )
            max_frame_dist_start = view_selector_config.get(
                "curriculum_start_max_frame_dist", 64
            )
            cur_min_frame_dist = int(
                min_frame_dist_start
                + (min_frame_dist - min_frame_dist_start) * progress
            )
            cur_max_frame_dist = int(
                max_frame_dist_start
                + (max_frame_dist - max_frame_dist_start) * progress
            )
            if cur_max_frame_dist <= cur_min_frame_dist:
                return None
            return two_frame_selector(
                frames, num_views, cur_min_frame_dist, cur_max_frame_dist
            )
        else:
            raise NotImplementedError(
                f"View selector type {view_selector_config.type} with curriculum {use_curriculum} is not implemented"
            )

    def __getitem__(self, idx):
        max_retries = 10
        for retry in range(max_retries):
            try:
                return self._load_scene(idx)
            except Exception as e:
                scene_path = (
                    self.all_scene_paths[idx].strip()
                    if idx < len(self.all_scene_paths)
                    else "unknown"
                )
                print(
                    f"[Dataset] Error loading scene {scene_path}: {e}. Retrying with another scene ({retry+1}/{max_retries})..."
                )
                idx = random.randint(0, len(self) - 1)
        raise RuntimeError(
            f"[Dataset] Failed to load any scene after {max_retries} retries"
        )

    def _load_scene(self, idx):
        scene_path = self.all_scene_paths[idx].strip()
        scene_root = scene_path.replace("opencv_cameras.json", "")
        data_json = json.load(open(scene_path, "r"))
        frames = data_json["frames"]
        scene_name = data_json["scene_name"]

        if self.inference and scene_name in self.view_idx_list:
            current_view_idx = self.view_idx_list[scene_name]
            image_indices = sorted(
                current_view_idx["context"] + current_view_idx["target"]
            )
            context_indices = torch.tensor(
                [image_indices.index(i) for i in current_view_idx["context"]]
            ).long()  # current_view_idx["context"]
            target_indices = torch.tensor(
                [image_indices.index(i) for i in current_view_idx["target"]]
            ).long()  # current_view_idx["target"]
        else:
            # sample input and target views
            image_indices = self.view_selector(frames, self.current_iteration)
            if image_indices is None or len(image_indices) == 0:
                return self.__getitem__(random.randint(0, len(self) - 1))
            context_indices = None
            target_indices = None

        image_paths_chosen = [frames[ic]["file_path"] for ic in image_indices]
        image_paths_chosen = [os.path.join(scene_root, it) for it in image_paths_chosen]
        frames_chosen = [frames[ic] for ic in image_indices]
        input_images, input_intrinsics, input_c2ws = self.preprocess_frames(
            frames_chosen, image_paths_chosen
        )

        # centerize and scale the poses (for unbounded scenes)
        scene_scale_factor = self.config.training.get("scene_scale_factor", 1.35)
        input_c2ws = self.preprocess_poses(input_c2ws, scene_scale_factor)

        image_indices = torch.tensor(image_indices).long().unsqueeze(-1)  # [v, 1]
        scene_indices = torch.full_like(image_indices, idx)  # [v, 1]
        indices = torch.cat([image_indices, scene_indices], dim=-1)  # [v, 2]

        data = {
            "image": input_images,
            "c2w": input_c2ws,  # not used in rayzer, loaded just following LVSM code
            "fxfycxcy": input_intrinsics,  # not used in rayzer, loaded just following LVSM code
            "index": indices,
            "scene_name": scene_name,
        }

        # used for evaluation
        if context_indices is not None:
            data["context_indices"] = context_indices
        if target_indices is not None:
            data["target_indices"] = target_indices

        return data


def two_frame_selector(frames, num_views, min_frame_dist, max_frame_dist):
    frame_dist = random.randint(min_frame_dist, max_frame_dist)
    if len(frames) <= frame_dist:
        return []
    start_frame = random.randint(0, len(frames) - frame_dist - 1)
    end_frame = start_frame + frame_dist
    rest_frames = random.sample(range(start_frame + 1, end_frame), num_views - 2)
    frame_indices = [start_frame] + rest_frames + [end_frame]
    return frame_indices
