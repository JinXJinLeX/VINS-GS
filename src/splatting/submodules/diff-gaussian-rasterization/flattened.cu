#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <memory>
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "cuda_rasterizer/auxiliary.h"


#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include <glm/glm.hpp>
#include "flattened.h"

BinningState BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain_(chunk, binning.point_list, P, 128);
	obtain_(chunk, binning.point_list_unsorted, P, 128);
	obtain_(chunk, binning.point_list_keys, P, 128);
	obtain_(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain_(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

__global__ void processflattenedCUDA(
    int P,
    int K,
    const float* xyzs,
    const int* xyz_ids,
    const float* scales,
    const float* rotations,
    const int* indexs,
    const float* normal,
    float* mean_ds,
    float* out_loss_size,
    float* out_loss_d,
    float* out_loss_normal)
{
    int idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

/*********************************************************** */
    if(indexs[K*idx]==indexs[K*idx+1])
    { 
        // printf("idx: %d, indexs[idx]: %d",idx, indexs[idx]);
        out_loss_normal[idx] = 0;
        out_loss_d[idx] = 0;
        out_loss_size[idx] = 0;
        return;
    }
    // printf("idx: %d, indexs[idx]: %d",idx, indexs[K*idx]);
    float3 xyz_cur = {xyzs[3*idx], xyzs[3*idx + 1], xyzs[3*idx + 2]};
    float4 rotation_cur = {rotations[4*idx], rotations[4*idx+1], rotations[4*idx+2], rotations[4*idx+3]};
    float3 scale_cur = {scales[3*idx], scales[3*idx+1], scales[3*idx+2]};

    float q_r = rotation_cur.x;
    float q_x = rotation_cur.y;
    float q_y = rotation_cur.z;
    float q_z = rotation_cur.w;
    float3 normal_cur = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};

    float3 normal_precomp = {normal[3*idx], normal[3*idx + 1], normal[3*idx + 2]};
    float cos_theta = normal_cur.x*normal_precomp.x + normal_cur.y*normal_precomp.y + normal_cur.z*normal_precomp.z;   
    if (cos_theta < 1 && cos_theta > 1 - 0.134)   // 0-30度
    {    
        out_loss_normal[idx] = 1 - cos_theta;
        for (int i = 1; i < K; i++)//一般是4
        {
            float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
            float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};
    
            float r = rotation.x;
            float x = rotation.y;
            float y = rotation.z;
            float z = rotation.w;
            float3 KNNnormal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};
    
            float cos_theta = normal_cur.x*KNNnormal.x + normal_cur.y*KNNnormal.y + normal_cur.z*KNNnormal.z;
            if (cos_theta < 1 && cos_theta > 1 - 0.134)   // 0-30度
            {
                mean_ds[idx] += xyz.x*KNNnormal.x + xyz.y*KNNnormal.y + xyz.z*KNNnormal.z;
            }
        }
        mean_ds[idx] /= K;
        float d = xyz_cur.x * normal_cur.x + xyz_cur.y * normal_cur.y + xyz_cur.z * normal_cur.z;
        out_loss_d[idx] += (d - mean_ds[idx]) * (d - mean_ds[idx]);
        
        out_loss_size[idx] = scale_cur.z;
    }
    else
    {
        out_loss_normal[idx] = 0;
        out_loss_d[idx] = 0;
        out_loss_size[idx] = scale_cur.z;
    }
/*********************************************************** */
}

__global__ void processflattenedBackwardCUDA(
    int P,
    int K,
    const float* xyzs,
    const int* xyz_ids,
    const float* scales,
    const float* rotations,
    const int* indexs,
    const float* normal,
    const float* mean_ds,
    const float* grad_out_loss_size,
    const float* grad_out_loss_d,
    const float* grad_out_loss_normal,
    float* dL_dxyzs,
    float* dL_dscales,
    float* dL_drotations)
{
   auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

/*********************************************************** */
    if(indexs[K*idx]==indexs[K*idx+1])
    { 
        return;
    }
    float3 xyz_cur = {xyzs[3*idx], xyzs[3*idx+1], xyzs[3*idx+2]};
    float4 rotation_cur = {rotations[4*idx], rotations[4*idx+1], rotations[4*idx+2], rotations[4*idx+3]};
    float3 scale_cur = {scales[3*idx], scales[3*idx+1], scales[3*idx+2]};

    float q_r = rotation_cur.x;
    float q_x = rotation_cur.y;
    float q_y = rotation_cur.z;
    float q_z = rotation_cur.w;
    float3 normal_cur = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};
    float3 normal_precomp = {normal[3*idx], normal[3*idx + 1], normal[3*idx + 2]};

    float cos_theta = normal_cur.x*normal_precomp.x + normal_cur.y*normal_precomp.y + normal_cur.z*normal_precomp.z;
    float dL_dout_loss_size = grad_out_loss_size[idx];
    float dL_dout_loss_d = grad_out_loss_d[idx];
    float dL_dout_loss_normal = grad_out_loss_normal[idx];
    // printf("dL_dout_loss_size %.8e \n",dL_dout_loss_size);
    // printf("dL_dout_loss_d %.8e \n",dL_dout_loss_d);
    // printf("dL_dout_loss_normal %.8e \n",dL_dout_loss_normal);

    float d = xyz_cur.x * normal_cur.x + xyz_cur.y * normal_cur.y + xyz_cur.z * normal_cur.z;

    if(cos_theta < 1 && cos_theta > 1 - 0.134)
    {
        //loss对法向量的导数
        float dL_dnx = - dL_dout_loss_normal*normal_precomp.x + dL_dout_loss_d*2*(d-mean_ds[idx])*xyz_cur.x;
        float dL_dny = - dL_dout_loss_normal*normal_precomp.y + dL_dout_loss_d*2*(d-mean_ds[idx])*xyz_cur.y;
        float dL_dnz = - dL_dout_loss_normal*normal_precomp.z + dL_dout_loss_d*2*(d-mean_ds[idx])*xyz_cur.z;

        atomicAdd(&(dL_dxyzs[3*idx]), dL_dout_loss_d* 2*(d-mean_ds[idx])*normal_cur.x);        //dL_dx_idx_i
        atomicAdd(&(dL_dxyzs[3*idx+1]), dL_dout_loss_d* 2*(d-mean_ds[idx])*normal_cur.y);      //dL_dy_idx_i
        atomicAdd(&(dL_dxyzs[3*idx+2]), dL_dout_loss_d* 2*(d-mean_ds[idx])*normal_cur.z);      //dL_dz_idx_i

        atomicAdd(&(dL_dscales[3*idx]), 0);        //dL_dx_idx_i
        atomicAdd(&(dL_dscales[3*idx+1]), 0);      //dL_dy_idx_i
        atomicAdd(&(dL_dscales[3*idx+2]), dL_dout_loss_size);      //dL_dz_idx_i

        atomicAdd(&(dL_drotations[4*idx]), (2 * q_y * dL_dnx - 2 * q_x * dL_dny));          //dL_drotation_w_current
        atomicAdd(&(dL_drotations[4*idx+1]), (2 * q_z * dL_dnx - 2 * q_r * dL_dny - 4 * q_x * dL_dnz));      //dL_drotation_x_current
        atomicAdd(&(dL_drotations[4*idx+2]), (2 * q_r * dL_dnx + 2 * q_z * dL_dny - 4 * q_y * dL_dnz));      //dL_drotation_y_current
        atomicAdd(&(dL_drotations[4*idx+3]), (2 * q_x * dL_dnx + 2 * q_y * dL_dny));      //dL_drotation_z_current
        // printf("dL_drotations: %f,%f,%f,%f",dL_drotations[4*idx],dL_drotations[4*idx+1],dL_drotations[4*idx+2],dL_drotations[4*idx+3]);
    }

/*********************************************************** */
}

__global__ void processelongatedCUDA(
    int P,
    int K,
    const float* xyzs,
    const int* xyz_ids,
    const float* rotations,
    const int* indexs,
    const float* direct,
    float* mean_ds,
    float* out_loss_d,
    float* out_loss_direct)
{
    int idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;
    if(indexs[K*idx]==indexs[K*idx+1])
    { 
        return;
    }
    float3 xyz_cur = {xyzs[3*idx], xyzs[3*idx+1], xyzs[3*idx+2]};
    float4 rotation_cur = {rotations[4*idx], rotations[4*idx+1], rotations[4*idx+2], rotations[4*idx+3]};
    float q_r = rotation_cur.x;
    float q_x = rotation_cur.y;
    float q_y = rotation_cur.z;
    float q_z = rotation_cur.w;

    float3 direct1 = {1-2*(q_y*q_y + q_z*q_z), 2*(q_x*q_y + q_r*q_z), 2*(q_x*q_z - q_r*q_y)};
    float3 direct2 = {2*(q_x*q_y - q_r*q_z), 1-2*(q_x*q_x + q_z*q_z), 2*(q_y*q_z + q_r*q_x)};
    float3 direct_cur;
    if((direct1.x*direct1.x+direct1.y*direct1.y+ direct1.z*direct1.z) < (direct2.x*direct2.x+direct2.y*direct2.y+ direct2.z*direct2.z))
    {
        direct_cur = {direct2.x, direct2.y, direct2.z};
    }
    else
    {
        direct_cur = {direct1.x, direct1.y, direct1.z};
    }

    float3 direct_precomp = {direct[3*idx], direct[3*idx + 1], direct[3*idx + 2]};
    float cos_theta = direct_cur.x*direct_precomp.x + direct_cur.y*direct_precomp.y + direct_cur.z*direct_precomp.z;   
    if (cos_theta < 1 && cos_theta > 1 - 0.134)   // 0-30度
    {    
        out_loss_direct[idx] = 1 - cos_theta;
    }
    else
    {
        out_loss_direct[idx] = 0;
        out_loss_d[idx] = 0;
    }
/********************************************************* */
    // float3 xyz_current = {xyzs[3*indexs[K*idx]], xyzs[3*indexs[K*idx] + 1], xyzs[3*indexs[K*idx] + 2]};
    // float4 rotation_current = {rotations[4*indexs[K*idx]], rotations[4*indexs[K*idx] + 1], rotations[4*indexs[K*idx] + 2], rotations[4*indexs[K*idx] + 3]};
    // float q_r = rotation_current.x;
    // float q_x = rotation_current.y;
    // float q_y = rotation_current.z;
    // float q_z = rotation_current.w;
    // float3 normal_current = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};

    // for (int i = 0; i < K; i++)
    // {
    //     float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
    //     float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

    //     float r = rotation.x;
    //     float x = rotation.y;
    //     float y = rotation.z;
    //     float z = rotation.w;
    //     float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};

    //     float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
    //     if (cos_theta < 1 && cos_theta > 1-0.4)   // 0-10度
    //     {
    //         mean_ds[indexs[K*idx]] += xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
    //         out_loss_normal[indexs[K*idx]] += 1 - cos_theta;
    //     }
    // }

    // mean_ds[indexs[K*idx]] /= K;

    // for (int i = 0; i < K; i++)
    // {
    //     float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
    //     float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

    //     float r = rotation.x;
    //     float x = rotation.y;
    //     float y = rotation.z;
    //     float z = rotation.w;
    //     float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};

    //     float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
    //     if (cos_theta < 1 && cos_theta > 1-0.004)   // 0-10度
    //     {
    //         float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
    //         out_loss_d[indexs[K*idx]] += (d - mean_ds[indexs[K*idx]]) * (d - mean_ds[indexs[K*idx]]);
    //     }

    // }
/********************************************************* */

}

__global__ void processelongatedBackwardCUDA(
    int P,
    int K,
    const float* xyzs,
    const int* xyz_ids,
    const float* rotations,
    const int* indexs,
    const float* mean_ds,
    // const float* grad_out_loss_size,
    const float* grad_out_loss_d,
    const float* grad_out_loss_normal,
    float* dL_dxyzs,
    float* dL_drotations)
{
   auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float4 rotation_current = {rotations[4*indexs[K*idx]], rotations[4*indexs[K*idx] + 1], rotations[4*indexs[K*idx] + 2], rotations[4*indexs[K*idx] + 3]};
    float q_r = rotation_current.x;
    float q_x = rotation_current.y;
    float q_y = rotation_current.z;
    float q_z = rotation_current.w;
    float3 normal_current = {2 * (q_x*q_z + q_r*q_y), 2 * (q_y*q_z - q_r*q_x), 1 - 2 * (q_x*q_x + q_y*q_y)};

    float dL_dout_loss_normal = grad_out_loss_normal[indexs[K*idx]];

    for (int i = 0; i < K; i++)
    {
        float3 xyz = {xyzs[3*indexs[K*idx+i]], xyzs[3*indexs[K*idx+i] + 1], xyzs[3*indexs[K*idx+i] + 2]};
        float4 rotation = {rotations[4*indexs[K*idx+i]], rotations[4*indexs[K*idx+i] + 1], rotations[4*indexs[K*idx+i] + 2], rotations[4*indexs[K*idx+i] + 3]};

        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float3 normal = {2 * (x*z + r*y), 2 * (y*z - r*x), 1 - 2 * (x*x + y*y)};


        float cos_theta = normal_current.x*normal.x + normal_current.y*normal.y + normal_current.z*normal.z;
        if(cos_theta < 1 && cos_theta > 1-0.004)
        {
            // depth_align 
            float d = xyz.x*normal.x + xyz.y*normal.y + xyz.z*normal.z;
            float dL_dout_loss_d = 2 * (d - mean_ds[indexs[K*idx]]) * grad_out_loss_d[indexs[K*idx]];
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i]]), dL_dout_loss_d * normal.x);          //dL_dx_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 1]), dL_dout_loss_d * normal.y);      //dL_dy_idx_i
            atomicAdd(&(dL_dxyzs[3*indexs[K*idx+i] + 2]), dL_dout_loss_d * normal.z);      //dL_dz_idx_i

            // depth_align + normal_align
            float dL_dnx = dL_dout_loss_d * xyz.x - normal_current.x;
            float dL_dny = dL_dout_loss_d * xyz.y - normal_current.y;
            float dL_dnz = dL_dout_loss_d * xyz.z - normal_current.z;

            atomicAdd(&(dL_drotations[4*indexs[K*idx+i]]), 2 * y * dL_dnx - 2 * x * dL_dny);          //dL_drotation_w_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 1]), 2 * z * dL_dnx - 2 * r * dL_dny - 4 * x * dL_dnz);      //dL_drotation_x_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 2]), 2 * r * dL_dnx + 2 * z * dL_dny - 4 * y * dL_dnz);      //dL_drotation_y_current
            atomicAdd(&(dL_drotations[4*indexs[K*idx+i] + 3]), 2 * x * dL_dnx + 2 * y * dL_dny);      //dL_drotation_z_current
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
flattened_align(
	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& knn_index,
    const torch::Tensor& normal)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);
    // printf("P %d\n", P);
    // printf("k %d\n", k);
    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);
    
    torch::Tensor mean_d = torch::full({P}, 0.0, float_opts);
    torch::Tensor out_loss_size = torch::full({P}, 0.0, float_opts).requires_grad_(true);
    torch::Tensor out_loss_d = torch::full({P}, 0.0, float_opts).requires_grad_(true);
    torch::Tensor out_loss_normal = torch::full({P}, 0.0, float_opts).requires_grad_(true);//初始化float类型的张量，大小是P
    
    torch::Device device(torch::kCUDA);//创建CUDA设备对象
    torch::TensorOptions options(torch::kByte);//设置张量数据类型为字节类型
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));//创建一个大小为0的空张量，分配在CUDA设备上，数据类型为字节型
    
    processflattenedCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
                                            xyzs.contiguous().data<float>(),
                                            xyz_ids.contiguous().data<int>(),
                                            scales.contiguous().data<float>(),
                                            rotations.contiguous().data<float>(),
                                            knn_index.contiguous().data<int>(),
                                            normal.contiguous().data<float>(),
                                            mean_d.contiguous().data<float>(),
                                            out_loss_size.contiguous().data<float>(),
                                            out_loss_d.contiguous().data<float>(),
                                            out_loss_normal.contiguous().data<float>());
    cudaDeviceSynchronize();
    return std::make_tuple(out_loss_size, out_loss_d, out_loss_normal, binningBuffer, mean_d);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
flattened_align_backward(
 	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& mean_d,
    const torch::Tensor& knn_index,
    const torch::Tensor& normal,
 	const torch::Tensor& grad_out_loss_size,
 	const torch::Tensor& grad_out_loss_d,
	const torch::Tensor& grad_out_loss_normal)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);

    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);

    torch::Tensor dL_dxyzs = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor dL_dscales = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor dL_drotations = torch::full({P, 4}, 0.0, float_opts);

    // printf("grad_out_loss_size %.8e\n",grad_out_loss_size);
    // printf("grad_out_loss_d %.8e\n",grad_out_loss_d);
    // printf("grad_out_loss_normal %.8e\n",grad_out_loss_normal);

    processflattenedBackwardCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
        xyzs.contiguous().data<float>(),
        xyz_ids.contiguous().data<int>(),
        scales.contiguous().data<float>(),
        rotations.contiguous().data<float>(),
        knn_index.contiguous().data<int>(),
        normal.contiguous().data<float>(),
        mean_d.contiguous().data<float>(),
        grad_out_loss_size.contiguous().data<float>(),
        grad_out_loss_d.contiguous().data<float>(),
        grad_out_loss_normal.contiguous().data<float>(),
        dL_dxyzs.contiguous().data<float>(),
        dL_dscales.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>());
    cudaDeviceSynchronize();

    constexpr int64_t kPrint = 10;
    int64_t actual = std::min<int64_t>(kPrint, grad_out_loss_size.size(0));

    auto gsz = grad_out_loss_size.slice(0, 0, actual).to(torch::kCPU);
    auto gd  = grad_out_loss_d.slice(0, 0, actual).to(torch::kCPU);
    auto gn  = grad_out_loss_normal.slice(0, 0, actual).to(torch::kCPU);

    auto g_xyz = dL_dxyzs.slice(0, 0, actual).to(torch::kCPU);
    auto g_sca = dL_dscales.slice(0, 0, actual).to(torch::kCPU);
    auto g_rot = dL_drotations.slice(0, 0, actual).to(torch::kCPU);
    printf("---- first %lld rows ----\n", actual);
    for (int64_t i = 0; i < actual; ++i)
    {
        printf("[%lld] size=% .8e  d=% .8e  normal=% .8e\n",i, gsz[i].item<float>(), gd[i].item<float>(), gn[i].item<float>());

        printf(" xyz=");
        for (int64_t s = 0; s < g_xyz.size(1); ++s) printf("% .6e ", g_xyz[i][s].item<float>());
        printf(" scale=");
        for (int64_t s = 0; s < g_sca.size(1); ++s) printf("% .6e ", g_sca[i][s].item<float>());
        printf(" rot=");
        for (int64_t r = 0; r < g_rot.size(1); ++r) printf("% .6e ", g_rot[i][r].item<float>());
        printf("\n");
    }

    return std::make_tuple(dL_dxyzs, dL_dscales, dL_drotations);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
elongated_align(
	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& knn_index,
    const torch::Tensor& direct)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);
    //     printf("P %d\n", P);
    //     printf("k %d\n", k);
    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);
    
    torch::Tensor mean_d = torch::full({P}, 0.0, float_opts);
    torch::Tensor out_loss_d = torch::full({P}, 0.0, float_opts);
    torch::Tensor out_loss_direct = torch::full({P}, 0.0, float_opts);
    
    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    
    processelongatedCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
                                            xyzs.contiguous().data<float>(),
                                            xyz_ids.contiguous().data<int>(),
                                            rotations.contiguous().data<float>(),
                                            knn_index.contiguous().data<int>(),
                                            direct.contiguous().data<float>(),
                                            mean_d.contiguous().data<float>(),
                                            out_loss_d.contiguous().data<float>(),
                                            out_loss_direct.contiguous().data<float>());
    cudaDeviceSynchronize();
    return std::make_tuple(out_loss_d, out_loss_direct, binningBuffer, mean_d);

    }

std::tuple<torch::Tensor, torch::Tensor>
elongated_align_backward(
 	const torch::Tensor& xyzs,
	const torch::Tensor& xyz_ids,
    const torch::Tensor& rotations,
    const torch::Tensor& binning_buffer,
    const torch::Tensor& mean_d,
    const torch::Tensor& knn_index,
 	const torch::Tensor& grad_out_loss_d,
	const torch::Tensor& grad_out_loss_direct)
{
    const int P = xyzs.size(0);
    const int k_P = knn_index.size(0);
    const int k = knn_index.size(1);

    auto int_opts = xyzs.options().dtype(torch::kInt32);
    auto float_opts = xyzs.options().dtype(torch::kFloat32);

    torch::Tensor dL_dxyzs = torch::full({P, 3}, 0.0, float_opts);
    torch::Tensor dL_drotations = torch::full({P, 4}, 0.0, float_opts);
    processelongatedBackwardCUDA<<<(k_P + 255) / 256, 256>>>(k_P, k,
        xyzs.contiguous().data<float>(),
        xyz_ids.contiguous().data<int>(),
        rotations.contiguous().data<float>(),
        knn_index.contiguous().data<int>(),
        mean_d.contiguous().data<float>(),
        grad_out_loss_d.contiguous().data<float>(),
        grad_out_loss_direct.contiguous().data<float>(),
        dL_dxyzs.contiguous().data<float>(),
        dL_drotations.contiguous().data<float>());
    cudaDeviceSynchronize();

    return std::make_tuple(dL_dxyzs, dL_drotations);
}