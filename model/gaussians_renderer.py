from typing import Union

import torch
import torch.nn as nn
from easydict import EasyDict as edict


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device=r.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L = R @ L
    return L


# SH coefficients
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]
C4 = [2.5033429417967046, -1.7701307697799304, 0.9461746957575601,
      -0.6690465435572892, 0.10578554691520431, -0.6690465435572892,
      0.47308734787878004, -1.7701307697799304, 0.6258357354491761]


def eval_sh(deg, sh, dirs):
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )
            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


class GaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.inv_scaling_activation = torch.log
        self.rotation_activation = torch.nn.functional.normalize
        self.opacity_activation = torch.sigmoid
        self.covariance_activation = build_covariance_from_scaling_rotation

    def __init__(self, sh_degree: int, scaling_modifier=None):
        self.sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        if self.sh_degree > 0:
            self._features_rest = torch.empty(0)
        else:
            self._features_rest = None
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.setup_functions()
        self.scaling_modifier = scaling_modifier

    def empty(self):
        self.__init__(self.sh_degree, self.scaling_modifier)

    def set_data(self, xyz, features, scaling, rotation, opacity):
        self._xyz = xyz
        self._features_dc = features[:, :1, :].contiguous()
        if self.sh_degree > 0:
            self._features_rest = features[:, 1:, :].contiguous()
        else:
            self._features_rest = None
        self._scaling = scaling
        self._rotation = rotation
        self._opacity = opacity
        return self

    def to(self, device):
        self._xyz = self._xyz.to(device)
        self._features_dc = self._features_dc.to(device)
        if self.sh_degree > 0:
            self._features_rest = self._features_rest.to(device)
        self._scaling = self._scaling.to(device)
        self._rotation = self._rotation.to(device)
        self._opacity = self._opacity.to(device)
        return self

    def filter(self, valid_mask):
        self._xyz = self._xyz[valid_mask]
        self._features_dc = self._features_dc[valid_mask]
        if self.sh_degree > 0:
            self._features_rest = self._features_rest[valid_mask]
        self._scaling = self._scaling[valid_mask]
        self._rotation = self._rotation[valid_mask]
        self._opacity = self._opacity[valid_mask]
        return self

    @property
    def get_scaling(self):
        if self.scaling_modifier is not None:
            return self.scaling_activation(self._scaling) * self.scaling_modifier
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        if self.sh_degree > 0:
            features_dc = self._features_dc
            features_rest = self._features_rest
            return torch.cat((features_dc, features_rest), dim=1)
        else:
            return self._features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )


def render_opencv_cam_gsplat(
    pc: GaussianModel,
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    sh_degree: Union[int, None] = None,
    near_plane=0.2,
    bg_color=(1.0, 1.0, 1.0),
):
    from gsplat import rasterization

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    bg_color = (
        torch.tensor(list(bg_color), dtype=torch.float32, device=C2W.device)
        .unsqueeze(0)
        .expand(C2W.size(0), -1)
    )
    W2C = C2W.inverse()

    intr = torch.zeros(
        fxfycxcy.size(0), 3, 3, device=fxfycxcy.device, dtype=fxfycxcy.dtype
    )
    intr[:, 0, 0] = fxfycxcy[:, 0]
    intr[:, 1, 1] = fxfycxcy[:, 1]
    intr[:, 0, 2] = fxfycxcy[:, 2]
    intr[:, 1, 2] = fxfycxcy[:, 3]
    intr[:, 2, 2] = 1.0

    render_colors, _, _ = rasterization(
        means3D, rotations, scales, opacity.squeeze(), shs,
        W2C, intr, width, height,
        near_plane=near_plane,
        sh_degree=sh_degree,
        backgrounds=bg_color,
        render_mode="RGB",
    )
    return {"render": render_colors.permute(0, 3, 1, 2)}


class Renderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sh_degree = config.model.gaussians.sh_degree
        self.gaussians_model = GaussianModel(config.model.gaussians.sh_degree)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self, xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy,
    ):
        """
        Args:
            xyz: [b, n_gaussians, 3]
            features: [b, n_gaussians, (sh_degree+1)^2, 3]
            scaling: [b, n_gaussians, 3]
            rotation: [b, n_gaussians, 4]
            opacity: [b, n_gaussians, 1]
            height, width: int
            C2W: [b, v, 4, 4]
            fxfycxcy: [b, v, 4]
        Returns:
            edict with render: [b, v, 3, height, width]
        """
        b, v = C2W.size(0), C2W.size(1)
        renderings = torch.zeros(
            b, v, 3, height, width, dtype=torch.float32, device=xyz.device
        )
        depth = torch.zeros(
            b, v, 1, height, width, dtype=torch.float32, device=xyz.device
        )
        alpha = torch.zeros(
            b, v, 1, height, width, dtype=torch.float32, device=xyz.device
        )

        for i in range(b):
            pc = self.gaussians_model.set_data(
                xyz[i], features[i], scaling[i], rotation[i], opacity[i]
            )
            near_plane = self.config.model.get("near_plane", 0.2)
            buffers = render_opencv_cam_gsplat(
                pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree,
                near_plane=near_plane,
            )
            renderings[i] = buffers["render"]

        return edict(render=renderings, depth=depth, alpha=alpha)


def get_point_range_func(gaussians_config):
    range_setting = gaussians_config.get(
        "range_setting", edict({"type": "object_centric_depth"})
    )
    print("range_setting:", range_setting)

    if range_setting.type == "object_centric_depth":
        return lambda t: (2.0 * torch.sigmoid(t) - 1.0) * 1.5 + 2.7
    elif range_setting.type == "linear_depth":
        near = range_setting.get("near", 0.0)
        far = range_setting.get("far", 500.0)
        return lambda t: torch.sigmoid(t) * (far - near) + near
    elif range_setting.type == "log_depth":
        near = range_setting.get("near", -6.2)
        far = range_setting.get("far", 6.2)
        return lambda t: torch.exp(torch.sigmoid(t) * (far - near) + near)
    elif range_setting.type == "disparity":
        near = range_setting.get("near", 0.1)
        far = range_setting.get("far", 500.0)
        return lambda t: 1.0 / (torch.sigmoid(t) * (1.0 / near - 1.0 / far) + 1.0 / far)
    else:
        raise NotImplementedError(f"Unknown range type: {range_setting.type}")


@torch.no_grad()
def build_stepback_c2ws(frame_c2ws: torch.Tensor, step_back_distance: float) -> torch.Tensor:
    R = frame_c2ws[..., :3, :3]
    t = frame_c2ws[..., :3, 3]
    z_world = R[..., :, 2]
    t_new = t - step_back_distance * z_world
    c2w_step = frame_c2ws.clone()
    c2w_step[..., :3, 3] = t_new
    return c2w_step
