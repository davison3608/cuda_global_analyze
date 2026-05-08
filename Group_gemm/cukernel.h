#pragma once
#include "cutorch.h"

/**
 * \param A_group    Array of pointers to A matrices (size: num_groups)
 * \param B_group    Array of pointers to B matrices (size: num_groups)
 * \param C_group    Array of pointers to C matrices (size: num_groups)
 * \param M          Array of M dimensions for each group (C: M x N)
 * \param N          Array of N dimensions for each group
 * \param K          Array of K dimensions for each group (A: M x K, B: K x N)
 * \param num_groups Total number of GEMM groups
 * \param alpha      Scalar alpha (broadcast to all groups)
 * \param beta       Scalar beta  (broadcast to all groups)
 * \brief 每个线程单数据持有 固定block=16x16 
 * \brief 每个gemm的二维网格展平后拼接到grid.x
*/
template<int BLOCK_X, int BLOCK_Y>
__global__ void group_gemm_merge_v1(
    const void* const A_group[], // [num_groups][M_i * K_i]
    const void* const B_group[], // [num_groups][K_i * N_i]
    void* C_group[], // [num_groups][M_i * N_i]
    const int* __restrict__ M_, // [num_groups]
    const int* __restrict__ N_, // [num_groups]
    const int* __restrict__ K_, // [num_groups]
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
);
extern "C" __host__ void group_gemm_merge_v1_launch(
    const void* A_group[], // [num_groups][M_i * K_i]
    const void* B_group[], // [num_groups][K_i * N_i]
    void* C_group[], // [num_groups][M_i * N_i]
    const int* M_host, // [num_groups]
    const int* N_host, // [num_groups]
    const int* K_host, // [num_groups]
    const int num_groups,
    float&alpha,
    float&beta,
    cudaStream_t&cu_str
) noexcept;

/**
 * \brief 每个线程2x2数据持有 固定block=16x16 
 * \brief 每个gemm的二维网格展平后拼接到grid.x
*/
template<int BLOCK_X, int BLOCK_Y, int Threads>
__global__ void group_gemm_merge_v2(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
);
extern "C" __host__ void group_gemm_merge_v2_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept;

/**
 * \brief 每个线程2x2数据持有 固定block=16x16 
 * \brief 每两个gemm的选出最大的二维网格后拼接到grid.x
 * \brief 每个二维网格处理两个gemm任务
 * 
 * \brief occupy与v2一致 说明仍然在资源限制内 
 * \brief 片段加载与内积warp停滞过大 堆积的mio事务过多
 * \brief gird尺寸缩小 L1与share延迟无法掩盖
*/
template<int BLOCK_X, int BLOCK_Y, int Threads>
__global__ void group_gemm_merge_v3_withpair(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ pair_grid_x,
    int* __restrict__ pair_grid_y
);
extern "C" __host__ void group_gemm_merge_v3_withpair_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept;

/**
 * \brief 每个线程2x2数据持有 固定block=16x16 
 * \brief 每个gemm的二维网格展平后拼接到grid.x
 * \brief half2加载 同时增加跨步K长度
*/
template<int BLOCK_X, int BLOCK_Y, int Threads, int Vec_len>
__global__ void group_gemm_merge_v3_vecload(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
);
extern "C" __host__ void group_gemm_merge_v3_vecload_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept;

/**
 * \brief 每个线程2x2数据持有 固定block=16x16 
 * \brief 每个gemm的二维网格展平后拼接到grid.x
 * \brief half2加载 同时增加跨步K长度
 * \brief 每个gemm任务引入双缓冲 
*/
template<int BLOCK_X, int BLOCK_Y, int Threads, int Vec_len>
__global__ void group_gemm_merge_v4_vecloadcache(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
);
extern "C" __host__ void group_gemm_merge_v4_vecloadcache_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept;

/**
 * \brief 每个线程2x2数据持有 固定block=16x16 
 * \brief 每个gemm的二维网格展平后拼接到grid.x
 * \brief half2加载 同时增加跨步K长度
 * \brief 每个gemm任务引入双缓冲 
 * \brief 乘积部分tensor core
*/
template<int BLOCK_X, int BLOCK_Y, int Threads, int Vec_len>
__global__ void group_gemm_merge_v5_vecloadcache_core(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
);
extern "C" __host__ void group_gemm_merge_v5_vecloadcache_core_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept;
