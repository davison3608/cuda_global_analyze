#include "../cudapacked.h"
//! 一维度线程块 一维网格
//! 网格线程尺寸覆盖到N 线程块尺寸应该大于线程块数量
//! 归约求和方式
template<int BLOCK_SIZE, int ELEMENTS>
__device__ __forceinline__ float rms_compute_addsum_reduce(float *volatile sm_input) 
{
    static_assert(ELEMENTS <= TIDSIZE);
    int blo_x=threadIdx.x;
    __shared__ float sm_tmp_sum[ELEMENTS];

    constexpr float _dot = 2.0f;
    if (blo_x < ELEMENTS)
    sm_tmp_sum[blo_x]=powf(sm_input[blo_x], _dot);
    else
    sm_tmp_sum[blo_x]=0.0f;
    __syncthreads();

    //块内所有线程跨步访问
    #pragma unroll
    for (int n=BLOCK_SIZE / 2; n>0; n=n / 2) {
    if (blo_x < n && (blo_x + n) < ELEMENTS) 
        sm_tmp_sum[blo_x]+=sm_tmp_sum[blo_x + n];
    //下次归约确保同步
    __syncthreads();
    }
    
    return sm_tmp_sum[0];
}
template<int BLOCK_SIZE, int BLOCK_NUM>
__global__ void rms_normalization_kernel_1(
    const float* __restrict__ input, float gamma, float beta, 
    float* __restrict__ output, int N, float eps,
    float *tmp_sum, float *tmp_blk_sum
) 
{
    static_assert(BLOCK_NUM <= BLOCK_SIZE);
    const int blk_x = blockIdx.x;   
    const int blo_x = threadIdx.x;  
    const int glo_x = blk_x * BLOCK_SIZE + blo_x;  

    __shared__ float sm_in[BLOCK_SIZE];
    //加载全局数据到共享内存
    if (glo_x >= N) 
        sm_in[blo_x]=0.0f;
    else 
        sm_in[blo_x]=input[glo_x];
    __syncthreads();

    //每个块计算局部和
    float blk_sum=rms_compute_addsum_reduce<BLOCK_SIZE, BLOCK_SIZE>(sm_in);
    //暂存到全局内存
    if (blo_x == 0)
        tmp_blk_sum[blk_x]=blk_sum;
    __threadfence();
    __syncthreads();

    //第一个块计算全局和
    float glo_sum=-INFINITY;
    if (blk_x == 0) {
    glo_sum=rms_compute_addsum_reduce<BLOCK_SIZE, BLOCK_NUM>(tmp_blk_sum);
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
