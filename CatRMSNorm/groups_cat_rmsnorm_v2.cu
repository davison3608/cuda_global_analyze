#include "cukernel.h"

template<int WARPSIZE>
__device__ __forceinline__ void warp_pow_sum_reduce(
    float val, float* sm_val,
    int&warp_id, int&lane_id
) 
{
    val = val * val;
    
    constexpr uint MASK_ALL = 0xffffffff;
    constexpr int Shfl_len0 = WARPSIZE / 2;
    constexpr int Shfl_len1 = Shfl_len0 / 2;
    constexpr int Shfl_len2 = Shfl_len1 / 2;
    constexpr int Shfl_len3 = Shfl_len2 / 2;
    constexpr int Shfl_len4 = Shfl_len3 / 2;
    
    val += __shfl_xor_sync(MASK_ALL, val, Shfl_len0);
    val += __shfl_xor_sync(MASK_ALL, val, Shfl_len1);
    val += __shfl_xor_sync(MASK_ALL, val, Shfl_len2);
    val += __shfl_xor_sync(MASK_ALL, val, Shfl_len3);
    val += __shfl_xor_sync(MASK_ALL, val, Shfl_len4);
    //冗余广播
    val = __shfl_sync(MASK_ALL, val, 0);
    //一次入到shared
    if (lane_id == 0)
        sm_val[warp_id] = val;
}

template<>
__global__ __launch_bounds__(512) void groups_cat_rmsnorm_v2
    <32, 16>(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim, float eps, 
    half* __restrict__ weights,
    half* __restrict__ res_out,
    float* __restrict__ tmp_variance
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCKSIZE = BLOCK_Y * BLOCK_X;
    constexpr int WARPSIZE = BLOCK_X;
    constexpr int WARPNUMS = BLOCK_Y * BLOCK_X / WARPSIZE;

    cooperative_groups::thread_block block_grp = cooperative_groups::this_thread_block();
    cooperative_groups::grid_group grid_grp = cooperative_groups::this_grid();
    int blo_x = block_grp.thread_index().x;
    int blo_y = block_grp.thread_index().y;
    int blk_x = grid_grp.block_index().x;
    
    int warp_id = blo_y;
    int lane_id = blo_x;
    int blo_idx = warp_id * WARPSIZE + lane_id;
    int glo_idx = blk_x * BLOCKSIZE + blo_idx;

    //线程持有数据
    float x_val {0.0f};
    float res_val {0.0f};
    float weight_val {0.0f};

    //线程越界掩码
    bool mask_in = false;
    mask_in = (glo_idx >= dim);

    x_val = mask_in? 0.0f : __half2float(x_in[glo_idx]);
    res_val = mask_in? 0.0f : __half2float(res_in[glo_idx]);
    weight_val = mask_in? 0.0f : __half2float(weights[glo_idx]);

    //残差连接
    float combined_val = x_val + res_val;
    float combined_val_ori = combined_val;

    __shared__ float sm_pow_sums[WARPNUMS];

    //每个warp平方归约和
    warp_pow_sum_reduce<WARPSIZE>(
        combined_val, sm_pow_sums,
        warp_id, lane_id
    );
    block_grp.sync();

    //每个首warp计算block总和
    if (warp_id == 0) {
    float warp_sum = (lane_id < WARPNUMS)? sm_pow_sums[lane_id]:0.0f;

    constexpr uint MASK_ALL = 0xffffffff;
    constexpr int Shfl_len0 = WARPSIZE / 2;
    constexpr int Shfl_len1 = Shfl_len0 / 2;
    constexpr int Shfl_len2 = Shfl_len1 / 2;
    constexpr int Shfl_len3 = Shfl_len2 / 2;
    constexpr int Shfl_len4 = Shfl_len3 / 2;
    
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len0);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len1);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len2);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len3);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len4);

    //0线程写入全局和
    if (lane_id == 0) {
        atomicAdd(tmp_variance, warp_sum);
        //保证全局线程可见
        __threadfence();
    }
    }
    //保证其他warp等待首warp完成 随后block统一读取全局数据
    block_grp.sync();

    //网格级别同步
    grid_grp.sync();

    //线程求均方值
    float variance = tmp_variance[0] / dim;
    variance = 1 / sqrtf(__fadd_rn(variance, eps));

    //归一化处理
    combined_val = __fmul_rn(combined_val, variance);

    //缩放权重处理
    combined_val = __fmul_rn(combined_val, weight_val);

    if (mask_in == false) {
        x_in[glo_idx] = __float2half_rn(combined_val);
        res_out[glo_idx] = __float2half_rn(combined_val_ori);
    }
    else
        return;
}
