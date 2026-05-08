#pragma once
#include "cutorch.h"

/**
 * \param gating_x            输入的linear门控分数 [N_tokens, n_routed_experts]
 * \param score_bias          门控偏置 [n_routed_experts]
 * \param N_tokens            token 数目（即batch size 接受>=1）
 * \param n_routed_experts    全局稀疏专家总数
 * \param topk_num            每个token最终选出的专家数（8）
 * \param num_groups          专家分组数（8）
 * \param topk_groups         每个token选出的top_groups数（4）
 * \param route_scale         选中权重缩放因子（用于sigmoid后缩放）
 * \param topk_weights        输出选中专家的权重 [N_tokens, topk_num]
 * \param topk_ids            输出 选中专家的索引 [N_tokens, topk_num]
 * \brief block启动x轴固定为warp长度 y轴固定等于num_groups (8)
 * \brief grid启动x轴等于N_tokens
 */
template<int BLOCK_X, int BLOCK_Y, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void GroupedTopkfused_fp16_v1(
    const half* __restrict__ gating_x,
    const float* __restrict__ score_bias,
    int N_tokens, int n_routed_experts,
    const int topk_num,
    const int num_groups,
    const int topk_groups,
    float route_scale,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids
);
template<int BLOCK_X, int BLOCK_Y, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void GroupedTopkfused_fp32_v1(
    const float* __restrict__ gating_x,
    const float* __restrict__ score_bias,
    int N_tokens, int n_routed_experts,
    const int topk_num,
    const int num_groups,
    const int topk_groups,
    float route_scale,
    float* __restrict__ topk_weights,
    int* __restrict__ topk_ids
);
extern "C" __host__ void GroupedTopkfused_v1_launch(
    const void* gating_x,
    const float* score_bias,
    int N_tokens, int n_routed_experts,
    int topk_num, int num_groups, int topk_groups,
    float route_scale,
    float* topk_weights,
    int* topk_ids,
    bool is_float16 
) noexcept;

