#include "cukernel.h"

template<>
__global__ void stage_combine::flash_attn_v2_combine_v1
    <16, 16>(
    const float* __restrict__ Attn_out, //[batch_size, num_q_heads, splitK, kv_lora_rank + 1]
    float* __restrict__ output, //[batch_size, num_q_heads, kv_lora_rank]
    int curr_seq_len, //实际序列长度
    int q_head_num,
    int kv_lora_rank,
    int splitk 
)
{
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;

    int blo_x = threadIdx.x;
    int blo_y = threadIdx.y;
    int blk_x = blockIdx.x;
    int blk_y = blockIdx.y;
    int blk_z = blockIdx.z;

    //线程对应的确切头维度位置与kv_lora_rank维度位置
    int cur_head = blk_y * BLOCK_Y + blo_y;
    int cur_kv = blk_x * BLOCK_X + blo_x;

    //当前批次
    int cur_batch = blk_z;

    int stride_mid_ob = q_head_num * splitk * (kv_lora_rank + 1); //attn_out batch步长
    int stride_mid_oh = splitk * (kv_lora_rank + 1); //attn_out head步长
    int stride_mid_os = kv_lora_rank + 1; //attn_out splitK步长
    int stride_obs = q_head_num * kv_lora_rank; //output batch 步长
    int&stride_oh = kv_lora_rank; //output head 步长

    if (cur_kv >= stride_oh || cur_head >= (stride_obs / stride_oh)) 
        return;
    if (cur_batch >= 1)
        return;

    //所有splitK累加 带有稳定参数
    float e_max = -INFINITY;
    float e_sum{0.0f};
    float acc_sum{0.0f};

    //计算基础偏移 当前批次的单头起始基准
    int base_head_offset = cur_batch * stride_mid_ob + cur_head * stride_mid_oh;

    #pragma unroll
    for (int k=0; k<splitk; k++) {
    //检查当前split_k有效范围
    int kv_len_per_split = (curr_seq_len + splitk - 1) / splitk;
    int split_start = kv_len_per_split * k;
    int split_end = min(split_start + kv_len_per_split, curr_seq_len);

    if (split_end > split_start) {
    //加载当前split_k的512维向量中的当前元素
    float tv = Attn_out[base_head_offset + k * stride_mid_os + cur_kv];            
    //加载当前split_k的logsumexp值
    float tlogic = Attn_out[base_head_offset + k * stride_mid_os + stride_oh];
    
    //数值稳定性处理
    float n_e_max = fmaxf(tlogic, e_max);
    float old_scale = expf(e_max - n_e_max);

    //缩放之前累积的结果
    acc_sum = __fmul_rn(acc_sum, old_scale);

    //累加当前split_k
    float exp_logic = __expf(tlogic - n_e_max);
    acc_sum += exp_logic * tv;

    //更新权重和
    e_sum = e_sum * old_scale + exp_logic;
    e_max = n_e_max;    
    }
    }

    //最终归一化并写入输出
    if (e_sum > 0.0f) {
    int output_offset = cur_batch * stride_obs + cur_head * stride_oh + cur_kv;
    output[output_offset] = __fdiv_rn(acc_sum, e_sum);
    }
}

extern "C" void stage_combine::v1_launch(
    float* Attn_out, float* output, int curr_seq_len, int q_head_num,
    int kv_lora_rank, int splitk 
) noexcept
{
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    dim3 block(BLOCK_X, BLOCK_Y);
    int grid_x = (kv_lora_rank + BLOCK_X - 1) / BLOCK_X;
    int num_heads = (q_head_num * kv_lora_rank) / kv_lora_rank;
    int grid_y = (num_heads + BLOCK_Y - 1) / BLOCK_Y;
    dim3 grid(grid_x, grid_y, 1);

    void* args[] = {
        (void*)&Attn_out, (void*)&output, (void*)&curr_seq_len, (void*)&q_head_num, 
        (void*)&kv_lora_rank, (void*)&splitk
    };

}
