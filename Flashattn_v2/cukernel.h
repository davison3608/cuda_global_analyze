#pragma once
#include "cutorch.h"

/**
 * flash attention flash v2版本 
 * 解码阶段q输入序列始终为1 并且固定batchszie等于1
 * kv按值头数目固定为1 GQA退化为MQA 因此广播kv_head_num = q_head_num
 * MLA下qk底秩(qk_head_dim) > v底秩(kv_lora_rank)
 * 区别于vllm实现内部无RoPe编码 未引入Paged attnetion
*/

namespace stage_decode {
/**
 * \param Q            [1, 1, q_head_num, qk_head_dim] 解码阶段当前token总为1
 * \param K            [1, max_seq_len, kv_head_num, qk_head_dim] 
 * \param V            [1, max_seq_len, kv_head_num, kv_lora_rank] 
 * \param Attn_out     [q_head_num, splitk, kv_lora_rank + 1] 中间注意力输出 最后一列为logsumexp
 * \param curr_seq_len 当前有效序列长度（≤ max_seq_len）
 * \param q_head_num   查询头数量（如 128）
 * \param qk_head_dim  查询/键头维度（如 576）
 * \param kv_lora_rank 值压缩维度 / LoRA 秩（如 512），也是输出特征维度
 * \param splitk       沿序列维度的分块数 curr_seq_len / splitk
 * \param sm_scale     qk缩放参数保持为qk_head_dim** -0.5f
 * \param logit_cap    logits 截断阈值（0 表示不截断）
 * 
 * \brief 固定block为32个warp 每个warp得到[1, 32]向量 block拼凑出qk[block_H, block_N, 32]
 * \brief 其中warpsize在qk_dim或是kv_dim上进行迭代处理
 * \brief grid启动x轴覆盖splitk y轴覆盖q_head_num
*/
template<int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
__global__ void flash_attn_v2_decode_v1(
    const half* __restrict__ Q, //[1, 1, q_head_num, qk_head_dim]
    const half* __restrict__ K, //[1, max_seq_len, kv_head_num, qk_head_dim]
    const half* __restrict__ V, //[1, max_seq_len, kv_head_num, kv_lora_rank]
    half* Attn_out, //[q_head_num, splitk, kv_lora_rank + 1]
    int curr_seq_len, 
    int q_head_num,
    int qk_head_dim,
    int kv_lora_rank,
    int splitk,
    float sm_scale,
    float logit_cap
);
extern "C" void v1_launch(
    half* Q, half* K, half* V, 
    half* Attn_out,
    int curr_seq_len, 
    int q_head_num,
    int qk_head_dim, 
    int kv_lora_rank,
    int splitk,
    float logit_cap
) noexcept;

/**
 * \brief 固定block为32个warp 每个block加载KV更长序列片段(BLOCK_N * 4) 因此curr_seq_len//splitk为4倍数
 * \brief 低秩上half2加载(qk_head_dim kv_lora_rank) 
 * \brief grid启动x轴覆盖splitk y轴覆盖q_head_num
*/
template<int BLOCK_X, int BLOCK_Y, int BLOCK_Z, int Vec_len>
__global__ void flash_attn_v2_decode_v2(
    const half* __restrict__ Q, 
    const half* __restrict__ K, 
    const half* __restrict__ V, 
    half* Attn_out, 
    int curr_seq_len, 
    int q_head_num,
    int qk_head_dim,
    int kv_lora_rank,
    int splitk,
    float sm_scale,
    float logit_cap
);
extern "C" void v2_launch(
    half* Q, half* K, half* V, 
    half* Attn_out,
    int curr_seq_len, 
    int q_head_num,
    int qk_head_dim, 
    int kv_lora_rank,
    int splitk,
    float logit_cap
) noexcept;


} // namespace stage_decode

namespace stage_combine {
/**
 * \param Attn_out     [splitk, q_head_num, kv_lora_rank + 1] 中间注意力输出 最后一列为logsumexp
 * \param output       [batch_size, num_q_heads, kv_lora_rank] 合并输出结果
 * \param curr_seq_len 当前有效序列长度（≤ max_seq_len）
*/
template<int BLOCK_X, int BLOCK_Y>
__global__ void flash_attn_v2_combine_v1(
    const float* __restrict__ Attn_out, //[batch_size, num_q_heads, splitK, kv_lora_rank + 1]
    float* __restrict__ output, //[batch_size, num_q_heads, kv_lora_rank]
    int curr_seq_len, //实际序列长度
    int q_head_num,
    int kv_lora_rank,
    int splitk 
);
extern "C" void v1_launch(
    float* Attn_out, 
    float* output,
    int curr_seq_len,
    int q_head_num,
    int kv_lora_rank,
    int splitk 
) noexcept;


} // namespace stage_combine
