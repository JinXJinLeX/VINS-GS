#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# xjl 20250901
import open3d as o3d
from scipy.spatial.transform import Rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        #xjl
        self.knn_tree = None
        self._knn_index = {}
        self._score = torch.empty(0)
        self._xyz_id = torch.empty(0)
        self.modify_id = []
        # self._type = torch.empty(0)
        # self._normal = torch.empty(0)
        # self._direct = torch.empty(0)
        self._sh_conf = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            #xjl
            self.knn_tree,
            self._knn_index,
            self._score,
            # self._type,
            # self._normal,
            # self._direct,
            self._sh_conf,

            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        #xjl
        # self._type,
        # self._normal,
        # self._direct,
        self._sh_conf,

        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_xyz_id(self):
        return self._xyz_id
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure
    
    #xjl 新添的属性#######################################################
    # @property
    # def get_type(self):
    #     return self._type
    
    # @property
    # def get_normal(self):
    #     return self._normal

    # @property
    # def get_direct(self):
    #     return self._direct
    
    @property
    def get_sh_conf(self):
        return self._sh_conf
    
    @staticmethod
    def eval_sh_bases(dirs: torch.Tensor, L=3) -> torch.Tensor:
        """
        计算实数球谐基函数 (up to L=3)
        Args:
            L: 最大阶数 (这里用 L=3)
            dirs: (N, 3) 单位向量方向 (x,y,z)
        Returns:
            Y: (N, (L+1)^2) SH 基函数值
        """
        assert L <= 3, "只实现到 L=3"
        x, y, z = dirs[:,0], dirs[:,1], dirs[:,2]
        N = dirs.shape[0]

        # 结果容器
        out = dirs.new_zeros((N, (L+1)**2))

        # --- L=0 ---
        out[:,0] = 0.28209479177387814  # sqrt(1/(4π))

        if L >= 1:
            out[:,1] = -0.4886025119029199 * y  # Y(1,-1)
            out[:,2] =  0.4886025119029199 * z  # Y(1,0)
            out[:,3] = -0.4886025119029199 * x  # Y(1,1)

        if L >= 2:
            out[:,4] =  1.0925484305920792 * x * y
            out[:,5] = -1.0925484305920792 * y * z
            out[:,6] =  0.31539156525252005 * (3*z*z - 1.0)
            out[:,7] = -1.0925484305920792 * x * z
            out[:,8] =  0.5462742152960396 * (x*x - y*y)

        if L >= 3:
            out[:,9]  = -0.5900435899266435 * y * (3*x*x - y*y)
            out[:,10] =  2.890611442640554 * x * y * z
            out[:,11] = -0.4570457994644658 * y * (5*z*z - 1.0)
            out[:,12] =  0.3731763325901154 * z * (5*z*z - 3.0)
            out[:,13] = -0.4570457994644658 * x * (5*z*z - 1.0)
            out[:,14] =  1.445305721320277 * z * (x*x - y*y)
            out[:,15] = -0.5900435899266435 * x * (x*x - 3*y*y)

        return out


        # 对可见高斯累积球谐(SH)置信度，避免同一高斯在同一帧被多次重复写入。
        
        # 参数
        # ----
        # gauss_ids : torch.Tensor (N,)
        #     当前帧所有可见高素的全局索引，可能重复（因为一个高素可能被多个像素看到）。
        # directions : torch.Tensor (N, 3)
        #     与 gauss_ids 一一对应的观测方向（单位向量），用于计算该视角下的 SH 基函数值。
        
        # 内部张量
        # --------
        # self._sh_conf : torch.Tensor (P, 16)
        #     每个高素在当前训练迭代的 SH 置信度累积量，16 = (L+1)^2，L=3。
        
        # 流程
        # ----
        # 1. 计算每条记录的 SH 基函数值 Y(N,16)。
        # 2. 对重复的 gauss_ids 做「唯一化」：只保留 uniq_ids，并用 inv 映射回原顺序。
        # 3. 先把 Y 的通道和 y_weight = Y.sum(1) 按 id 累加，得到每个 uniq 高素的「权重总和」。
        # 4. 再把权重乘回对应的 SH 基 Y[inv]，用 index_add_ 一次性写入 self._sh_conf。
        # 这样每个 uniq 高素只写一次，显著降低显存峰值与原子写冲突。

        # directions : (N,3)  从 Gaussian 中心指向相机中心的向量（未归一化）
        # eps        : 手动调节的距离平滑量，越小则越近权重越高
    def update_sh_conf(self, gauss_ids, directions, eps=0.5):

        d = directions.norm(dim=1, keepdim=True).clamp_min(1e-8)  # (N,1)
        unit_dir = directions / d                                 # 单位方向
        w = 1.0 / (d + eps)                                       # 手动权重

        Y = self.eval_sh_bases(unit_dir, 3)                       # (N,16)
        Y *= w                                                    # 加权

        with torch.no_grad():
            uniq_ids, inv = torch.unique(gauss_ids, return_inverse=True)
            self._sh_conf.index_add_(0, uniq_ids, Y[inv])

        # # 1. 计算模长并防止除 0
        # dir_norm = directions.norm(dim=1, keepdim=True).clamp_min(1e-8)
        # # 2. 权重 = 1 / 模长
        # weight = 1.0 / dir_norm                # (N,1)
        # # 3. 带权的“方向”向量（不再是单位向量）
        # weighted_dir = weight * directions     # (N,3)

        # # 4. 用加权后的方向计算 SH 基
        # Y = self.eval_sh_bases(weighted_dir, 3)   # (N,16)

        # print("Y.shape",Y.shape)
        # print("_sh_conf.shape",self._sh_conf.shape)
        # with torch.no_grad():
        #     uniq_ids, inv = torch.unique(gauss_ids, return_inverse=True)
        #     print("uniq_ids.shape",uniq_ids.shape)
        #     y_weight = torch.zeros_like(uniq_ids, dtype=torch.float32).index_add_(0, inv, Y.sum(1))  # 按 id 累加权重
        #     self._sh_conf.index_add_(0, uniq_ids, y_weight.unsqueeze(1) * Y[inv])  # 再乘回基

    def query_conf(self, gauss_ids, directions):
        Y = self.eval_sh_bases(directions, 3)  # (N,16)
        conf = (self._sh_conf[gauss_ids] * Y).sum(dim=-1)
        return conf
############################################################################
    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    #xjl
    #修改了其中初始化部分
    # ply
    # format ascii 1.0
    # element vertex 1625439
    # property float x
    # property float y
    # property float z
    # property uchar red
    # property uchar green
    # property uchar blue
    # property float type
    # property float dx
    # property float dy
    # property float dz
    # property float nx
    # property float ny
    # property float nz
    # end_header

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda() #将点云坐标转换为CUDA张量
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda()) #将RGB颜色转换为球谐函数系数
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda() #初始化特征张量，形状为(点数, 3, (最大球谐度+1)²)
        features[:, :3, 0 ] = fused_color #分别设置颜色特征和其余特征的初始值，初始色彩为RGB颜色，高阶为0
        features[:, 3:, 1:] = 0.0

        # xjl 20250929 初始化置信度的球谐函数 (L=3 → 16维)
        sh_conf = torch.zeros((fused_point_cloud.shape[0], 16), dtype=torch.float, device="cuda")
        sh_conf[:, 0] = 0.0  # 初始各向同性
        self._sh_conf = sh_conf


        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1 #此处的旋转矩阵没有法向量，是单位矩阵

        #xjl 根据type来进行初始化
        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # scales = torch.sqrt(dist2)[..., None]
        # zero = torch.full_like(scales, 1e-3)
        # scales = torch.log(torch.concat([scales, scales, zero], dim=1))
        # rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots[:, 0] = 1

        # self._type = torch.tensor(np.asarray(pcd.type)).float().cuda()
        # # type_point_maks_0 = (self._type == 0).squeeze() #平坦
        # type_point_maks_1 = (self._type == 1).squeeze() #尖锐
        # type_point_maks_2 = (self._type == 2).squeeze() #兼有
        # type_point_maks_3 = (self._type == 3).squeeze() #普通
        # scales[:, 2][type_point_maks_3] = scales[:, 0][type_point_maks_3] #普通
        # scales[:, 1][type_point_maks_3] = scales[:, 0][type_point_maks_3] #普通
        # scales[:, 1][type_point_maks_1] = scales[:, 2][type_point_maks_1] #尖锐
        # scales[:, 1][type_point_maks_2] = scales[:, 2][type_point_maks_2] #兼有
        # self._normal = torch.tensor(np.asarray(pcd.normals)).float().cuda()
        # self._direct = torch.tensor(np.asarray(pcd.directs)).float().cuda()

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        # xjl 20250906
        # self.reset_xyz_id()

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # xjl 20250930 新增字段
        # l.append('type')                                    # 1 列
        for i in range(self._sh_conf.shape[1]):             # 16 列
            l.append('sh_conf_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        #xjl 20250930
        sh_conf = self._sh_conf.detach().cpu().numpy()
        # type = self._type.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, type, sh_conf), axis=1)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, sh_conf), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # xjl  读取 16 维 SH-confidence
        shconf_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sh_conf_")]
        shconf_names = sorted(shconf_names, key=lambda x: int(x.split('_')[-1]))
        sh_conf = np.zeros((xyz.shape[0], len(shconf_names)))   # (N, 16)
        for idx, attr_name in enumerate(shconf_names):
            sh_conf[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # >>>>>>  end  <<<<<<

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # xjl  把置信度存进 self._sh_conf
        self._sh_conf = torch.tensor(sh_conf, dtype=torch.float, device="cuda")
        # >>>>>>  end  <<<<<<
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        # xjl 20250906
        self._xyz_id = self._xyz_id[valid_points_mask]
        # self._type = self._type[valid_points_mask]
        # self._normal = self._normal[valid_points_mask]
        # self._direct = self._direct[valid_points_mask]
        self._sh_conf = self._sh_conf[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_type, new_normal, new_direct, new_sh_conf):
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_sh_conf):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        # "type": new_type}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # xjl 20250906
        self._xyz_id = torch.concat([self._xyz_id, new_xyz_id], dim=0)
        # self._type = torch.concat([self._type, new_type], dim=0)
        # self._normal = torch.concat([self._normal, new_normal], dim=0)
        # self._direct = torch.concat([self._direct, new_direct], dim=0)
        self._sh_conf = torch.concat([self._sh_conf, new_sh_conf], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        # xjl 20250906
        new_xyz_id = self._xyz_id[selected_pts_mask].repeat(N)
        # new_type = self._type[selected_pts_mask].repeat(N, 1)
        # new_normal = self._normal[selected_pts_mask].repeat(N,1)
        # new_direct = self._direct[selected_pts_mask].repeat(N,1)
        new_sh_conf = self._sh_conf[selected_pts_mask].repeat(N,1)

        self.modify_id.extend(self._xyz_id[selected_pts_mask].tolist())
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_type, new_normal, new_direct, new_sh_conf)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_sh_conf)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        # xjl 20250906
        new_xyz_id = self._xyz_id[selected_pts_mask]
        # new_type = self._type[selected_pts_mask]
        # new_normal = self._normal[selected_pts_mask]
        # new_direct = self._direct[selected_pts_mask]
        new_sh_conf = self._sh_conf[selected_pts_mask]

        # selected_pts_mask = torch.logical_and(selected_pts_mask, self._type.squeeze() == 1)
        # new_xyz[new_type.squeeze() == 1] += self.position_gradient_accum[selected_pts_mask] / self.denom[selected_pts_mask]
        self.modify_id.extend(new_xyz_id.tolist())

        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_type, new_normal, new_direct, new_sh_conf)
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_xyz_id, new_sh_conf)

    def gaussian_iou_3d(self, mu1, s1, R1, mu2, s2, R2):
        """
        近似计算两个 3D 高斯的 IoU = vol(intersect)/vol(union)。
        把每个高斯沿主轴切成 3σ 的立方体，求立方体 IoU，再乘一个高斯核衰减系数。
        返回 0~1 标量 tensor。
        """
        # 3σ 半轴长
        a1, a2 = 3*s1, 3*s2   # [3]
        # 世界空间 3σ 立方体 min/max
        corners1 = torch.stack([-a1, a1], dim=1)  # [3,2]
        corners2 = torch.stack([-a2, a2], dim=1)
        bbox1_min = mu1 + (R1 @ corners1[:, 0:1]).squeeze()
        bbox1_max = mu1 + (R1 @ corners1[:, 1:2]).squeeze()
        bbox2_min = mu2 + (R2 @ corners2[:, 0:1]).squeeze()
        bbox2_max = mu2 + (R2 @ corners2[:, 1:2]).squeeze()

        # 相交区间
        inter_min = torch.maximum(bbox1_min, bbox2_min)
        inter_max = torch.minimum(bbox1_max, bbox2_max)
        inter = torch.clamp(inter_max - inter_min, min=0)
        vol_inter = inter.prod()

        vol1 = (2*a1).prod()
        vol2 = (2*a2).prod()
        vol_union = vol1 + vol2 - vol_inter + 1e-12
        iou_box = vol_inter / vol_union

        # 中心距离衰减
        d = torch.linalg.norm(mu1 - mu2)
        sigma_avg = (s1.mean() + s2.mean())*0.5
        decay = torch.exp(-(d**2)/(2*(sigma_avg**2)+1e-7))
        return iou_box * decay

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii

        # xjl 20250906
        self.reset_xyz_id()
        self.modify_id = []

        # 致密化
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # xjl 20250901
    def createKDTree(self, pc):
        # build KDTree
        self.knn_tree = o3d.geometry.KDTreeFlann(pc)

    # xjl 20250907添加knn的条件，只计算type一致的点
    # def findtypeKNN(self, target_types=0, k=4):
    #     # t1 = time.time()

    #     points_np = self.get_xyz.detach().cpu().numpy()
    #     types_np = self._type.detach().cpu().numpy() #把点云的type转为数组
    #     pc = o3d.geometry.PointCloud()
    #     pc.points = o3d.utility.Vector3dVector(points_np)

    #     self.createKDTree(pc)
    #     # 创建掩码，筛选type
    #     type_mask = np.isin(types_np.squeeze(), target_types)
    #     target_points = points_np[type_mask]
    #     # nn
    #     knn_indices = []
    #     for p in target_points:
    #         _, indices, _ = self.knn_tree.search_knn_vector_3d(p, knn=k)
    #         knn_indices.append(indices)
    #     self._knn_index[k] = knn_indices
    #     # to torch tensor
    #     data = [torch.tensor(np.array(index, dtype=np.int32)).unsqueeze(0) for index in self._knn_index[k]]
    #     data = torch.concat(data, dim=0).cuda()
    #     # t2 = time.time()
    #     # print('\nknn time(s) : ', f'{t2 - t1:.3f}')
    #     return data

    def findtypeKNN(self, target_types=[0,2], k=4):
        points_np = self.get_xyz.detach().cpu().numpy()          # [n, 3]
        types_np  = self._type.detach().cpu().numpy().squeeze()  # [n]

        # 1. 全部点建 KD-Tree（保证返回全局索引）
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_np)
        self.createKDTree(pc)

        # 2. 只把 type=0 的点坐标和全局序号抽出来
        mask0   = np.isin(types_np, target_types)        # bool [n]
        pts0    = points_np[mask0]                       # 仅用于查询
        idx0    = np.where(mask0)[0]                     # 对应的全局下标

        # 3. 对每个全局点搜 4 个最近 type=0 点
        n = points_np.shape[0]
        knn_mat = np.empty((n, k), dtype=np.int32)

        for i in range(n):
            _, idx_local, _ = self.knn_tree.search_knn_vector_3d(points_np[i], k)
            # idx_local 已经是全局号，但需过滤非 type=0,2 的点（极少情况才会出现）
            # 保险起见，再取前 4 个 type=0 的
            idx_type0 = [x for x in idx_local if mask0[x]][:k]
            # 不足 4 个用自身补
            while len(idx_type0) < k:
                idx_type0.append(i)
            knn_mat[i] = idx_type0

        # 4. 转 Tensor
        self._knn_index[k] = torch.from_numpy(knn_mat).cuda()
        return self._knn_index[k]

    def findKNN(self, k=4):
        # t1 = time.time()
        points_np = self.get_xyz.detach().cpu().numpy()

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_np)

        self.createKDTree(pc)

        # 原版的nn
        self._knn_index[k] = [self.knn_tree.search_knn_vector_3d(p, knn=k)[1] for p in pc.points]

        # to torch tensor
        data = [torch.tensor(np.array(index, dtype=np.int32)).unsqueeze(0) for index in self._knn_index[k]]
        data = torch.concat(data, dim=0).cuda()
        # t2 = time.time()
        # print('\nknn time(s) : ', f'{t2 - t1:.3f}')
        return data

    def computeNormal(self, k=30):

        return None
        # build point
        t_1 = time.time()
        points_np = self.get_xyz.detach().cpu().numpy()
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_np)

        self.createKDTree(pc)

        # 寻找最近邻点
        list_k = [2, k]
        for _k in list_k:
            self._knn_index[_k] = [self.knn_tree.search_knn_vector_3d(p, knn=_k)[1] for p in pc.points]

        # 转换为torch张量
        data = [self.get_xyz[index, :].unsqueeze(0) for index in self._knn_index[k]]
        data = torch.concat(data, dim=0)

        # 计算法向量
        mean = torch.mean(data, dim=1, keepdim=True)
        # 减去均值
        centered_data = data - mean.expand_as(data)
        # 计算协方差矩阵
        covariance_matrix = torch.bmm(centered_data.permute(0, 2, 1), centered_data) / k
        # 使用svd函数进行奇异值分解
        U, S, _ = torch.linalg.svd(covariance_matrix)

        print(f'{time.time()-t_1} s')
        return U[:, :, 2]

    def reset_xyz_id(self):
        number_point = self._xyz.shape[0]
        index_id = np.arange(number_point)
        self._xyz_id = torch.tensor(index_id).int().cuda()
