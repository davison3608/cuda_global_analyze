#pragma once
#include "cutorch.h"

#ifdef __HIP_PLATFORM_AMD__

#define __shfl_xor_sync(mask, val, offset_shfl) __shfl_xor(val, offset_shfl)
#define __shfl_sync(mask, val, srclane) __shfl(val, srclane)
#define __syncwarp(mask) //rocm warp默认同步

#endif

/**
 * x_in 输入原始特征向量[dim] half类型
 * res_in 输入残差向量[dim] half类型
 * dim 特征向量维度
 * eps 数值稳定项 防止分母为0
 * weights 逐维度缩放权重[dim] float类型
 * res_out 输出残差中间值[dim] half类型
 * \brief 块级并行 block[512,] grid[16,]
*/
template<int BLOCK_X, int GRID_X>
__global__ void groups_cat_rmsnorm_v1(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim, float eps, 
    half* __restrict__ weights,
    half* __restrict__ res_out
);

/**
 * x_in 输入原始特征向量[dim] half类型
 * res_in 输入残差向量[dim] half类型
 * dim 特征向量维度
 * eps 数值稳定项 防止分母为0
 * weights 逐维度缩放权重[dim] float类型
 * res_out 输出残差中间值[dim] half类型
 * \brief warp级并行 block[32, 16] 
*/
template<int BLOCK_X, int BLOCK_Y>
__global__ void groups_cat_rmsnorm_v2(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim, float eps, 
    half* __restrict__ weights,
    half* __restrict__ res_out,
    float* __restrict__ tmp_variance
);

/**
 * x_in 输入原始特征向量[dim] half类型
 * res_in 输入残差向量[dim] half类型
 * dim 特征向量维度
 * eps 数值稳定项 防止分母为0
 * weights 逐维度缩放权重[dim] float类型
 * res_out 输出残差中间值[dim] half类型
 * \brief warp级并行 float4向量化处理 block[32, 16] 
 * \brief 存在精度偏差 并且额外的冗余向量线程判断产生停滞
*/
template<int BLOCK_X, int BLOCK_Y, int Vec_len>
__global__ void groups_cat_rmsnorm_v3(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim, float eps, 
    half* __restrict__ weights,
    half* __restrict__ res_out,
    float* __restrict__ tmp_variance
);

/**
 * x_in 输入原始特征向量[dim] half类型
 * res_in 输入残差向量[dim] half类型
 * dim 特征向量维度
 * eps 数值稳定项 防止分母为0
 * weights 逐维度缩放权重[dim] float类型
 * res_out 输出残差中间值[dim] half类型
 * \brief warp级并行 block[32, 16] 
*/
template<int BLOCK_X, int BLOCK_Y>
void groups_cat_rmsnorm_v4(
    half* x_in,
    half* res_in,
    int dim, float eps, 
    half* weights,
    half* res_out,
    cudaStream_t&cu_str
);


