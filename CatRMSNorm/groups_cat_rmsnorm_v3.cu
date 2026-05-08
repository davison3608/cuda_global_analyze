#include "cukernel.h"

template<int WARPSIZE>
__device__ __forceinline__ void warp_pow_sum_reduce(
    float val, float* sm_val,
    int&warp_id, int&lane_id
) 
{    
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
    val = __shfl_sync(MASK_ALL, val, 0);

    if (lane_id == 0)
        sm_val[warp_id] = val;
}

template<>
__global__ __launch_bounds__(512) void groups_cat_rmsnorm_v3
    <32, 16, 4>(
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
    constexpr int Vec_len = 4;
    constexpr int Res_len = Vec_len / 2;

    cooperative_groups::thread_block block_grp = cooperative_groups::this_thread_block();
    cooperative_groups::grid_group grid_grp = cooperative_groups::this_grid();
    int blo_x = block_grp.thread_index().x;
    int blo_y = block_grp.thread_index().y;
    int blk_x = grid_grp.block_index().x;
    
    int warp_id = blo_y;
    int lane_id = blo_x;
    int blo_idx = block_grp.thread_rank();
    int glo_idx = grid_grp.thread_rank();

    half2* x_in2 = reinterpret_cast<half2*>(x_in);
    half2* res_in2 = reinterpret_cast<half2*>(res_in);
    half2* weights2 = reinterpret_cast<half2*>(weights);

    //线程持有数据
    float4 x_val_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 res_val_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 wes_val_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    int dim_vec = dim / Vec_len;
    int dim_rem = dim % Vec_len;

    //向量越界掩码 线程越界掩码
    bool mask_vec = false; 
    bool mask_in = false;

    mask_vec = (glo_idx >= dim_vec);
    mask_in = (glo_idx >= (dim_vec + (dim_rem>0? 1:0))); 

    if (!mask_vec && !mask_in) {
        int vec_offset = Res_len * glo_idx;
        float2 x_val_tmp0 = __half22float2(x_in2[vec_offset + 0]);
        float2 x_val_tmp1 = __half22float2(x_in2[vec_offset + 1]);
        float2 res_val_tmp0 = __half22float2(res_in2[vec_offset + 0]);
        float2 res_val_tmp1 = __half22float2(res_in2[vec_offset + 1]);
        float2 wes_val_tmp0 = __half22float2(weights2[vec_offset + 0]);
        float2 wes_val_tmp1 = __half22float2(weights2[vec_offset + 1]);

        x_val_vec.x = x_val_tmp0.x;
        x_val_vec.y = x_val_tmp0.y;
        x_val_vec.z = x_val_tmp1.x;
        x_val_vec.w = x_val_tmp1.y;
        res_val_vec.x = res_val_tmp0.x;
        res_val_vec.y = res_val_tmp0.y;
        res_val_vec.z = res_val_tmp1.x;
        res_val_vec.w = res_val_tmp1.y;
        wes_val_vec.x = wes_val_tmp0.x;
        wes_val_vec.y = wes_val_tmp0.y;
        wes_val_vec.z = wes_val_tmp1.x;
        wes_val_vec.w = wes_val_tmp1.y;
    }
    else if (mask_vec && !mask_in) {
    int elem_start = glo_idx * Vec_len; 

    #pragma unroll
    for (int i = 0; i < dim_rem; i++) {
        int global_idx = elem_start + i;

        //按单个half加载
        half x_h = x_in[global_idx];
        half res_h = res_in[global_idx];
        half w_h = weights[global_idx];

        switch(i) {
        case 0: 
        x_val_vec.x = __half2float(x_h); 
        res_val_vec.x = __half2float(res_h); 
        wes_val_vec.x = __half2float(w_h); 
            break;
        case 1: 
        x_val_vec.y = __half2float(x_h); 
        res_val_vec.y = __half2float(res_h); 
        wes_val_vec.y = __half2float(w_h); 
            break;
        case 2: 
        x_val_vec.z = __half2float(x_h); 
        res_val_vec.z = __half2float(res_h); 
        wes_val_vec.z = __half2float(w_h); 
            break;
        case 3: 
        x_val_vec.w = __half2float(x_h); 
        res_val_vec.w = __half2float(res_h); 
        wes_val_vec.w = __half2float(w_h); 
            break;
        }
    }
    }
    else
        return;

    //向量残差连接    
    float4 combined_val_vec = make_float4(
        x_val_vec.x + res_val_vec.x,
        x_val_vec.y + res_val_vec.y,
        x_val_vec.z + res_val_vec.z,
        x_val_vec.w + res_val_vec.w
    );
    float4 combined_val_vec_ori = combined_val_vec;

    __shared__ float sm_pow_sums[WARPNUMS];

    //每个warp平方归约和
    float combined_val_sum = __fmul_rn(combined_val_vec.x, combined_val_vec.x) +
        __fmul_rn(combined_val_vec.y, combined_val_vec.y) +
        __fmul_rn(combined_val_vec.z, combined_val_vec.z) + 
        __fmul_rn(combined_val_vec.w, combined_val_vec.w);
    
    warp_pow_sum_reduce<WARPSIZE>(
        combined_val_sum, sm_pow_sums,
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
    combined_val_vec.x = __fmul_rn(combined_val_vec.x, variance);
    combined_val_vec.y = __fmul_rn(combined_val_vec.y, variance);
    combined_val_vec.z = __fmul_rn(combined_val_vec.z, variance);
    combined_val_vec.w = __fmul_rn(combined_val_vec.w, variance);

    //缩放权重处理
    combined_val_vec.x = __fmul_rn(combined_val_vec.x, wes_val_vec.x);
    combined_val_vec.y = __fmul_rn(combined_val_vec.y, wes_val_vec.y);
    combined_val_vec.z = __fmul_rn(combined_val_vec.z, wes_val_vec.z);
    combined_val_vec.w = __fmul_rn(combined_val_vec.w, wes_val_vec.w);

    half2* res_out2 = reinterpret_cast<half2*>(res_out);

    if (!mask_vec && !mask_in) {
    int vec_offset = Res_len * glo_idx;
    x_in2[vec_offset + 0] = __float22half2_rn(
        make_float2(combined_val_vec.x, combined_val_vec.y)
    );
    x_in2[vec_offset + 1] = __float22half2_rn(
        make_float2(combined_val_vec.z, combined_val_vec.w)
    );
    res_out2[vec_offset + 0] = __float22half2_rn(
        make_float2(combined_val_vec_ori.x, combined_val_vec_ori.y)
    );
    res_out2[vec_offset + 1] = __float22half2_rn(
        make_float2(combined_val_vec_ori.z, combined_val_vec_ori.w)
    );
    }
    else {
    int elem_start = glo_idx * Vec_len; 

    #pragma unroll
    for (int i = 0; i < dim_rem; i++) {
        int global_idx = elem_start + i;

        switch(i) {
        case 0: 
        x_in[global_idx] = __float2half(combined_val_vec.x);
        res_out[global_idx] = __float2half(combined_val_vec_ori.x);
            break;
        case 1: 
        x_in[global_idx] = __float2half(combined_val_vec.y);
        res_out[global_idx] = __float2half(combined_val_vec_ori.y);
            break;
        case 2: 
        x_in[global_idx] = __float2half(combined_val_vec.z);
        res_out[global_idx] = __float2half(combined_val_vec_ori.z);
            break;
        case 3: 
        x_in[global_idx] = __float2half(combined_val_vec.w);
        res_out[global_idx] = __float2half(combined_val_vec_ori.w);
            break;
        }
    }
    }
}
