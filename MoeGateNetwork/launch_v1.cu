#include "cukernel.h"

extern "C" __host__ void GroupedTopkfused_v1_launch(
    const void* gating_x,
    const float* score_bias,
    int N_tokens, int n_routed_experts,
    int topk_num, int num_groups, int topk_groups,
    float route_scale,
    float* topk_weights,
    int* topk_ids,
    bool is_float16 
) noexcept
{
    constexpr int BLOCK_X = 32; //warp == 32/64
    constexpr int BLOCK_Y = 8; //8 == num_groups
    constexpr int BLOCK_SIZE = BLOCK_Y * BLOCK_X;
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(N_tokens, 1); //N_tokens >= 1

    void* args[] = {
        &gating_x, &score_bias, 
        &N_tokens, &n_routed_experts,
        &topk_num, &num_groups, &topk_groups,
        &route_scale, 
        &topk_weights,
        &topk_ids
    };
    if (is_float16)
    CUDA_CHECK(cudaLaunchKernel(
        (void*)&GroupedTopkfused_fp16_v1<BLOCK_X, BLOCK_Y, BLOCK_SIZE>,
        grid, block, args,
        0, cudaStreamDefault
    ));
    else
    CUDA_CHECK(cudaLaunchKernel(
        (void*)&GroupedTopkfused_fp32_v1<BLOCK_X, BLOCK_Y, BLOCK_SIZE>,
        grid, block, args,
        0, cudaStreamDefault
    ));
    return;
}
