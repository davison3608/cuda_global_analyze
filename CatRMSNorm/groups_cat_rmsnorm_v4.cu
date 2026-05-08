#include "cukernel.h"

template<int BLOCK_X, int BLOCK_Y>
__global__ void groups_cat_rmsnorm_v4_reduce_warp(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim,
    float* __restrict__ tmp_variance,
    half* __restrict__ res_out
)
{
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
    int blo_idx = block_grp.thread_rank();
    int glo_idx = grid_grp.thread_rank();

    //线程持有数据
    float x_val {0.0f};
    float res_val {0.0f};

    //线程越界掩码
    bool mask_in = false;
    mask_in = (glo_idx >= dim);

    x_val = mask_in? 0.0f : __half2float(x_in[glo_idx]);
    res_val = mask_in? 0.0f : __half2float(res_in[glo_idx]);

    //残差连接
    float combined_val = x_val + res_val;
    float combined_val_ori = combined_val;

    __shared__ float sm_pow_sums[WARPNUMS];

    float val = combined_val;
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
    val = __shfl_sync(MASK_ALL, val, 0);

    if (lane_id == 0)
        sm_pow_sums[warp_id] = val;
    block_grp.sync();

    float block_sum {0.0f};
    //每个首warp计算block总和
    if (warp_id == 0) {
    float warp_sum = (lane_id < WARPNUMS)? sm_pow_sums[lane_id]:0.0f;
    
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len0);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len1);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len2);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len3);
    warp_sum += __shfl_xor_sync(MASK_ALL, warp_sum, Shfl_len4);

    //0线程写入全局和
    if (lane_id == 0) 
        block_sum = warp_sum;
    }

    if (mask_in == false) {
        res_out[glo_idx] = __float2half_rn(combined_val_ori);
    
        if (warp_id == 0 && lane_id == 0)
        tmp_variance[blk_x] = block_sum;
    }
}

template<int BLOCK_X, int BLOCK_Y>
__global__ void groups_cat_rmsnorm_v4_rms(
    half* __restrict__ res_out,
    float* __restrict__ tmp_variance,
    int dim, float eps,
    half* __restrict__ weights,
    half* __restrict__ x_in
)
{
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
    int blo_idx = block_grp.thread_rank();
    int glo_idx = grid_grp.thread_rank();
    
    //线程持有数据
    float combined_val {0.0f};
    float variance_val {0.0f};
    float weight_val {0.0f};

    //线程越界掩码 block平方和越界掩码
    bool mask_in = false;
    bool mask_tmp = false;
    mask_in = glo_idx >= dim;
    mask_tmp = lane_id >= BLOCK_Y;

    combined_val = mask_in? 0.0f:__half2float(res_out[glo_idx]);
    variance_val = mask_tmp? 0.0f:tmp_variance[lane_id];
    weight_val = mask_in? 0.0f:__half2float(weights[glo_idx]);

    float val = variance_val;
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
    variance_val = __shfl_sync(MASK_ALL, val, 0);

    //线程求均方值
    float variance = variance_val / dim;
    variance = 1 / sqrtf(__fadd_rn(variance, eps));

    //归一化处理
    combined_val = __fmul_rn(combined_val, variance);

    //缩放权重处理
    combined_val = __fmul_rn(combined_val, weight_val);

    if (mask_in == false) 
        x_in[glo_idx] = __float2half_rn(combined_val);
    else
        return;
}

template<>
void groups_cat_rmsnorm_v4<32, 16>(
    half* x_in,
    half* res_in,
    int dim, float eps, 
    half* weights,
    half* res_out,
    cudaStream_t&cu_str
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCKSIZE = BLOCK_Y * BLOCK_X;
    dim3 block(BLOCK_X, BLOCK_Y);
    int GRID_X = (dim + BLOCKSIZE - 1) / BLOCKSIZE;
    int GRID_Y = 1;
    dim3 grid(GRID_X, GRID_Y);
    
    float* tmp_variance = nullptr;
    ssize_t tmp_size = (GRID_X + 1) * sizeof(float);
    cudaMallocAsync(&tmp_variance, tmp_size, cu_str);
    cudaMemsetAsync(tmp_variance, 0.0f, tmp_size, cu_str);

    void* kernelArgs_0[] = {
        &x_in, &res_in, &dim,  
        &tmp_variance, &res_out
    };
    void* kernelArgs_1[] = {
        &res_out, &tmp_variance, &dim, &eps,
        &weights, &x_in
    };

    CUDA_CHECK(cudaStreamSynchronize(cu_str));
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v4_reduce_warp<BLOCK_X, BLOCK_Y>,
        grid, block, 
        kernelArgs_0, 
        0, cu_str
    );
    CUDA_CHECK(cudaStreamSynchronize(cu_str));
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v4_rms<BLOCK_X, BLOCK_Y>,
        grid, block, 
        kernelArgs_1, 
        0, cu_str
    );
    CUDA_CHECK(cudaStreamSynchronize(cu_str));
    return;
}
