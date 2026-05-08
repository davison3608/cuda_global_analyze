#pragma once
#include "cutorch.h"

namespace gptj_style {
/**
 * \param cos_sin_cache 预计算缓存cos_sin_cache
 * \param positions     每个token在序列中的绝对位置索引 当前为decode阶段序列长度为1 [batch_size, seq_len=1]
 * \param query         查询张量(不连续依赖跨步访问) 默认头数量为32 [batch_size, seq_len=1, num_heads=32, dim]
 * \param key           键张量(不连续依赖跨步访问) 默认头数目为1 [batch_size, seq_len=1, num_heads=1, dim]
 * \param q_seq_stride  q的seqlen维度的跨步 num_heads * dim
 * \param q_head_stride q的num_heads维度的跨步 1 * dim
 * \param k_seq_stride  k的seqlen维度的跨步 num_heads * dim
 * \param k_head_stride k的num_heads维度的跨步 1 * dim
 * \param seq_len       序列长度 token数目 保证等于1
 * \param num_heads     query张量头数目 保证为16倍数
 * \param rotary_dim    实际参与编码的维度大小 通常等于head_size 保证为32倍数
 * \brief 固定batchsize等于1
 * \brief block固定启动32/64 x 16 grid启动x轴等于seq_len
*/
template<int BLOCK_X, int BLOCK_Y, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void rope_withyarn_gptj_v1(
    half* __restrict__ cos_sin_cache, //[max_position_embeddings, rotary_dim]
    int* __restrict__ positions,      //[batch_size, seq_len=1]
    half* __restrict__ query,         //[batch_size, seq_len=1, num_heads=32, dim]
    half* __restrict__ key,           //[batch_size, seq_len=1, num_heads=1, dim]
    int q_seq_stride, int q_head_stride,  
    int k_seq_stride, int k_head_stride, 
    int seq_len, int num_heads,                        
    int rotary_dim
);
extern "C" __host__ void rope_withyarn_gptj_v1_launch(
    half* cos_sin_cache, int* positions, half* query, half* key,     
    int q_seq_stride, int q_head_stride, 
    int k_seq_stride, int k_head_stride,    
    int seq_len, int num_heads, int rotary_dim
) noexcept;
    
} // namespace gptj_style

namespace neox_style {
/**
 * \param cos_sin_cache 预计算缓存cos_sin_cache
 * \param positions     每个token在序列中的绝对位置索引 当前为decode阶段序列长度为1 [batch_size, seq_len=1]
 * \param query         查询张量(不连续依赖跨步访问) 默认头数量为32 [batch_size, seq_len=1, num_heads=32, dim]
 * \param key           键张量(不连续依赖跨步访问) 默认头数目为1 [batch_size, seq_len=1, num_heads=1, dim]
 * \param q_seq_stride  q的seqlen维度的跨步 num_heads * dim
 * \param q_head_stride q的num_heads维度的跨步 1 * dim
 * \param k_seq_stride  k的seqlen维度的跨步 num_heads * dim
 * \param k_head_stride k的num_heads维度的跨步 1 * dim
 * \param seq_len       序列长度 token数目 保证等于1
 * \param num_heads     query张量头数目 保证为16倍数
 * \param rotary_dim    实际参与编码的维度大小 通常等于head_size 保证为32倍数
 * \brief 固定batchsize等于1
 * \brief block固定启动32/64 x 16 grid启动x轴等于seq_len
*/
template<int BLOCK_X, int BLOCK_Y, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE) void rope_withyarn_neox_v1(
    half* __restrict__ cos_sin_cache, //[max_position_embeddings, rotary_dim]
    int* __restrict__ positions,      //[batch_size, seq_len=1]
    half* __restrict__ query,         //[batch_size, seq_len=1, num_heads=32, dim]
    half* __restrict__ key,           //[batch_size, seq_len=1, num_heads=1, dim]
    int q_seq_stride, int q_head_stride,  
    int k_seq_stride, int k_head_stride, 
    int seq_len, int num_heads,                        
    int rotary_dim
);
extern "C" __host__ void rope_withyarn_neox_v1_launch(
    half* cos_sin_cache, int* positions, half* query, half* key,     
    int q_seq_stride, int q_head_stride, 
    int k_seq_stride, int k_head_stride, 
    int seq_len, int num_heads, int rotary_dim
) noexcept;

} // namespace neox_style
