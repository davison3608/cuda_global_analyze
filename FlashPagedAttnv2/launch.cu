#include "cukernel.h"

using namespace decode;
using namespace combine;

extern "C" __host__ void flash_paged_attnv2_v1_launch(
    float* Query, 
    float* K_cache,
    float* V_cache,
    int* block_table,  
    int num_q_heads, 
    int qk_head_dim, 
    int kv_lora_rank,
    int seq_lens, int splitk,
    float sm_scale,                 
    float logits,                   
    float* attn_out  
) noexcept
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCK_SZ = BLOCK_Y * BLOCK_X;
    const dim3 block(BLOCK_X, BLOCK_Y);

    constexpr int BLOCK_H = 1;
    constexpr int BLOCK_N = BLOCK_Y * 1;
    dim3 grid(splitk, num_q_heads / BLOCK_H);

    int num_kv_head = 1;
    void* args[] = {
        &Query, &K_cache, &V_cache, &block_table,
        &num_q_heads, &num_kv_head, &qk_head_dim, &kv_lora_rank,
        &seq_lens, &splitk,
        &sm_scale, &logits,
        &attn_out
    };
    void* func = (void*)&flash_paged_attnv2_v1<
        BLOCK_X, BLOCK_Y, 1, BLOCK_SZ, 
        BLOCK_N, BLOCK_H
    >;
    CUDA_CHECK(cudaLaunchKernel(
        func, grid, block,
        args, 0, cudaStreamDefault
    ));
    return;
}

extern "C" __host__ void flash_paged_attnv2_v2_launch(
    float* Query, 
    float* K_cache,
    float* V_cache,
    int* block_table,  
    int num_q_heads, 
    int qk_head_dim, 
    int kv_lora_rank,
    int seq_lens, int splitk,
    float sm_scale,                 
    float logits,                   
    float* attn_out  
) noexcept
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCK_SZ = BLOCK_Y * BLOCK_X;
    const dim3 block(BLOCK_X, BLOCK_Y);

    constexpr int BLOCK_H = 1;
    constexpr int BLOCK_N = BLOCK_Y * 1;
    dim3 grid(splitk, num_q_heads / BLOCK_H);

    int num_kv_head = 1;
    void* args[] = {
        &Query, &K_cache, &V_cache, &block_table,
        &num_q_heads, &num_kv_head, &qk_head_dim, &kv_lora_rank,
        &seq_lens, &splitk,
        &sm_scale, &logits,
        &attn_out
    };
    void* func = (void*)&flash_paged_attnv2_v2<
        BLOCK_X, BLOCK_Y, 1, BLOCK_SZ, 
        BLOCK_N, BLOCK_H
    >;
    CUDA_CHECK(cudaLaunchKernel(
        func, grid, block,
        args, 0, cudaStreamDefault
    ));
    return;
}
