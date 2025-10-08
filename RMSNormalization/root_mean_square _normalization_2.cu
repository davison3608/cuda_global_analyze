#include "../cudapacked.h"
//! 一维线程块 一维网格
//! 网格线程尺寸覆盖到N
//! warp归约 保证WR_NUM <= 32
template<int ELEMENTS>
__device__ __forceinline__ float rms_compute_addsum_reduce_sf(float *volatile sm_input) 
{
    constexpr int   WRSIZE = 32;
    constexpr int   MASK   = 0xFFFFFFFF;
    constexpr float _dot   = 2.0f;
    static_assert(ELEMENTS <= WRSIZE);
    int blo_x=threadIdx.x;

    float sum=0.0f;
    if (blo_x < ELEMENTS)
        sum=powf(sm_input[blo_x], _dot);
    __syncwarp(MASK);

    #pragma unroll
    for (int i=WRSIZE / 2; i>0; i=i / 2) {
    if (blo_x < i && (blo_x + i) < ELEMENTS)
    sum=sum + __shfl_xor_sync(MASK, sum, i, WRSIZE);
    //无需__syncwarp
    }
    //0线程广播
    sum=__shfl_sync(MASK, sum, 0, WRSIZE);
    return sum;
}
template<int WR_NUM>
__global__ void rms_normalization_kernel_2(
    const float* __restrict__ input, float gamma, float beta, 
    float* __restrict__ output, int N, float eps,
    float *tmp_sum, float *tmp_blk_sum
) 
{
    constexpr int WRSIZE = 32;
    static_assert(WR_NUM <= WRSIZE);
    const int blk_x = blockIdx.x;   
    const int blo_x = threadIdx.x;  
    const int glo_x = blk_x * WRSIZE + blo_x;  

    __shared__ float sm_in[WRSIZE];
    //加载全局数据到共享内存
    if (glo_x >= N) 
        sm_in[blo_x]=0.0f;
    else 
        sm_in[blo_x]=input[glo_x];
    __syncthreads();

    //每个块计算局部和
    float blk_sum=rms_compute_addsum_reduce_sf<WRSIZE>(sm_in);
    //暂存到全局内存
    if (blo_x == 0)
        tmp_blk_sum[blk_x]=blk_sum;
    __threadfence();
    __syncthreads();

    //第一个块计算全局和
    float glo_sum=-INFINITY;
    if (blk_x == 0) {
    glo_sum=rms_compute_addsum_reduce_sf<WR_NUM>(tmp_blk_sum);
    //广播到所有块
    if (blo_x == 0)
        *tmp_sum=glo_sum;
    }
    __threadfence();  
    __syncthreads();

    glo_sum = *tmp_sum;
    __syncthreads();

    //计算rms变量
    float rms = sqrtf((glo_sum / N) + eps);

    //归一化后写回
    if (glo_x >= N) 
        return;
    else {
    float xi=sm_in[blo_x] / rms;
    output[glo_x]=xi * gamma + beta;
    }
}
