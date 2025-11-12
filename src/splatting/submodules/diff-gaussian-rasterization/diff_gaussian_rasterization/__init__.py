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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

# xjl_20250903
# 平坦型高斯对齐
class _flattened_align(torch.autograd.Function):
    @staticmethod
    def forward(ctx_, xyz, xyz_id, scales, rotation, knn_index, normal):
        loss_size, loss_d, loss_normal, binning_buffer, mean_d = _C.flattened_align(xyz, xyz_id, scales, rotation, knn_index, normal)
        # Keep relevant tensors for backward
        ctx_.save_for_backward(xyz, xyz_id, scales, rotation, binning_buffer, knn_index, mean_d, normal)

        return loss_size, loss_d, loss_normal

    @staticmethod
    def backward(ctx_, grad_out_loss_size, grad_out_loss_d, grad_out_loss_normal):

        xyz, xyz_id, scales, rotation, binning_buffer, knn_index, mean_d, normal= ctx_.saved_tensors #提前存好的中间量

        grad_xyz, grad_scales, grad_rotation = _C.flattened_align_backward(xyz, xyz_id, scales, rotation, binning_buffer, mean_d, knn_index, normal, grad_out_loss_size, grad_out_loss_d, grad_out_loss_normal)

        grads = (
            grad_xyz,
            None,
            grad_scales,
            grad_rotation,
            None,
            None
        )

        return grads

class Flattened_Align(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, xyz, xyz_id, scales, rotation, knn_index, normal):
        return _flattened_align.apply(xyz, xyz_id, scales, rotation, knn_index, normal)
    
# 细长型高斯对齐
class _elongated_align(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, xyz_id, rotation, knn_index, direct):
        loss_d, loss_normal, binning_buffer, mean_d = _C.elongated_align(xyz, xyz_id, rotation, knn_index, direct)

        # Keep relevant tensors for backward
        ctx.save_for_backward(xyz, xyz_id, rotation, binning_buffer, knn_index, mean_d)
        return loss_d, loss_normal

    @staticmethod
    def backward(ctx, grad_out_loss_d, grad_out_loss_normal):
        xyz, xyz_id, rotation, binning_buffer, knn_index, mean_d = ctx.saved_tensors

        grad_xyz, grad_rotation = _C.elongated_align_backward(xyz, xyz_id, rotation, binning_buffer, mean_d, knn_index, grad_out_loss_d, grad_out_loss_normal)

        grads = (
            grad_xyz,
            None,
            grad_rotation,
            None
        )

        return grads


class Elongated_Align(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, xyz_id, rotation, knn_index, direct):
        return _elongated_align.apply(xyz, xyz_id, rotation, knn_index, direct)
def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    sh_conf,
    # confidence,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        sh_conf,
        # confidence,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        sh_conf,
        # confidence,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            sh_conf,
            # confidence,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, out_confidence = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)
        return color, radii, invdepths, out_confidence

    @staticmethod
    def backward(ctx, grad_out_color, _1, grad_out_depth, _2):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color,
                grad_out_depth, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.antialiasing,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)        

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            None,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None, sh_conf = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])
        if sh_conf is None:
            sh_conf = torch.Tensor([])
        # if confidence is None:
        #     confidence = torch.Tensor([])
        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            sh_conf
            # confidence
        )

