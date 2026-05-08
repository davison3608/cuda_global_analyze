#include "cukernel.h"

__managed__ float tmp_variance {0.0f};

template<>
__global__ void groups_cat_rmsnorm_v1
    <512, 16>(
    half* __restrict__ x_in,
    half* __restrict__ res_in,
    int dim, float eps, 
    half* __restrict__ weights,
    half* __restrict__ res_out
)
{
    constexpr int BLOCK_X = 512;
    constexpr int GRID_X = 16;
    
    int blo_x = threadIdx.x;
    int blk_x = blockIdx.x;

    //线程持有数据
    __shared__ float sm_x_in[BLOCK_X];
    __shared__ float sm_x_in_ori[BLOCK_X];
    float src_res_in {0.0f};
    float src_wes_in {0.0f};

    if ((blk_x * BLOCK_X + blo_x) < dim) {
        src_res_in = __half2float(res_in[blk_x * BLOCK_X + blo_x]);
        src_wes_in = __half2float(weights[blk_x * BLOCK_X + blo_x]);
        float x_val = __half2float(x_in[blk_x * BLOCK_X + blo_x]);
        sm_x_in[blo_x] = __powf(x_val + src_res_in, 2.0f);
        sm_x_in_ori[blo_x] = x_val + src_res_in;
    }
    else {
        sm_x_in[blo_x] = 0.0f;
        sm_x_in_ori[blo_x] = 0.0f;
    }
    __syncthreads();

    //block归约和
    float blo_sum {0.0f};
    if ((blk_x * BLOCK_X + blo_x) < dim) 
        blo_sum = sm_x_in[blo_x];

    #pragma unroll
    for (int len = BLOCK_X; len>0; len/=2) {
        if (blo_x < len) {
        blo_sum += sm_x_in[blo_x + len]; 
        sm_x_in[blo_x] = blo_sum; 
        }
        __syncthreads(); 
    }

    //每个block首线程求和
    if (blo_x == 0 && blk_x * BLOCK_X < dim)
        atomicAdd(&tmp_variance, blo_sum);
    __threadfence();
    __syncthreads();

    //线程求均方值
    float variance = tmp_variance / dim;

    //归一化处理
    variance = 1 / sqrtf(__fadd_rn(variance, eps));
    float x_out = __fmul_rn(sm_x_in_ori[blo_x], variance); 

    //缩放权重处理
    x_out = __fmul_rn(x_out, src_wes_in);

    //残差部分
    float x_res = __fadd_rn(sm_x_in_ori[blo_x], src_res_in);

    if ((blk_x * BLOCK_X + blo_x) < dim) {
        res_out[blk_x * BLOCK_X + blo_x] = __float2half_rn(x_res);
        x_in[blk_x * BLOCK_X + blo_x] = __float2half_rn(x_out);
    }
    else
        return;
}
