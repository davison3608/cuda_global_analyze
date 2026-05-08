#pragma once

#ifdef __HIP_PLATFORM_AMD__

#define __shfl_xor_sync(mask, var, lane_mask) __shfl_xor(var, lane_mask)
#define __shfl_sync(mask, var, lane) __shfl(var, lane)
#define __syncwarp(mask) //默认同步

#define __hmul_rn(a, b) __hmul(a, b)
#define __hadd_rn(a, b) __hadd(a, b)

#endif

namespace decode {
/**
 * \brief 基于MLA MQA flashattnv2版本的decode阶段 支持pagedattn
 * 
 * \param Query 输入Q在decode阶段seqlen始终为1[1, 1, num_q_heads, qk_head_dim]
 * \param PagedCahce KV形状均为[1, seq_lens, 1, dim]并分散在缓存页中
 * \param block_table 长度为blockn组数目储存分页缓存索引
 * \param attn_out 中间注意力输出
 * 
 * \brief block(32, 16, 1) warpsize对应dim低维 16warpnums负责序列切片blockn blockh头切片为1
 * \brief grid(splitk, num_q_heads // BLOCK_H) 每一行所有block对应一个splitk分组计算
 * \brief block针对注意力矩阵tile[BLOCK_H, BLOCK_N]进行online softmax
 * 维护局部最值[BLOCK_H, ] 累计指数和[BLOCK_H, ] 累加输出[BLOCK_H, kv_lora_rank]
 * 
 * \brief 对于每个tile先得到局部最值m'tile与全局局部最值比较得到m'new同时保留全局旧最值m'old
 * \brief 旧状态缩放因子exp(m'old-m'new)  当前局部缩放因子exp(m'tile-m'new)
 * \brief 对于累加指数和 l'new = l'old​ x exp(m'old-m'new) + l'tile x exp(m'tile-m'new)
 * \brief 对于累加输出 acc'new = acc'old x exp(m'old-m'new) + acc'tile x exp(m'tile-m'new)
 * \brief combine阶段的logsumexp依赖第一阶段的每个splitk局部维护数据
*/
template<
    int BLOCK_X, int BLOCK_Y, int BLOCK_Z, int BLOCK_SZ, 
    int BLOCK_N, int BLOCK_H
>
__global__ __launch_bounds__(BLOCK_SZ)  void flash_paged_attnv2_v1(
    float* __restrict__ Query,      //[1, 1, num_q_heads, qk_head_dim]
    float* __restrict__ K_cache,    //[N_blocks, BLOCK_N, num_kv_heads, qk_head_dim]
    float* __restrict__ V_cache,    //[N_blocks, BLOCK_N, num_kv_heads, kv_lora_rank] 
    int* __restrict__ block_table,  //[seq_lens // BLOCK_N]
    int num_q_heads,                //query头数目
    int num_kv_head,                //MQA下始终为1
    int qk_head_dim,                //qk低维长度
    int kv_lora_rank,               //v低维长度
    int seq_lens, int splitk,       //有效tokens长度与分片数目
    float sm_scale,                 //缩放参数
    float logits,                   //tanh裁剪
    float* __restrict__ attn_out    //[1, splitk, num_q_heads, 2 + kv_lora_rank]
);
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
) noexcept;

/**
 * \brief qk注意力矩阵计算与加权计算引入缓冲
*/
template<
    int BLOCK_X, int BLOCK_Y, int BLOCK_Z, int BLOCK_SZ, 
    int BLOCK_N, int BLOCK_H
>
__global__ __launch_bounds__(BLOCK_SZ)  void flash_paged_attnv2_v2(
    float* __restrict__ Query,      //[1, 1, num_q_heads, qk_head_dim]
    float* __restrict__ K_cache,    //[N_blocks, BLOCK_N, num_kv_heads, qk_head_dim]
    float* __restrict__ V_cache,    //[N_blocks, BLOCK_N, num_kv_heads, kv_lora_rank] 
    int* __restrict__ block_table,  //[seq_lens // BLOCK_N]
    int num_q_heads,                //query头数目
    int num_kv_head,                //MQA下始终为1
    int qk_head_dim,                //qk低维长度
    int kv_lora_rank,               //v低维长度
    int seq_lens, int splitk,       //有效tokens长度与分片数目
    float sm_scale,                 //缩放参数
    float logits,                   //tanh裁剪
    float* __restrict__ attn_out    //[1, splitk, num_q_heads, 2 + kv_lora_rank]
);
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
) noexcept;

    
} // namespace decode

namespace combine {
    
} // namespace combine


