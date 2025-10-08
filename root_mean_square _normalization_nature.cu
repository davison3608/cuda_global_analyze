#include "../cudapacked.h"
//! 一维度线程块 一维网格
//! 网格线程尺寸覆盖到N 线程块尺寸应该大于线程块数量
//! 原子求和方式
template<int BLOCK_SIZE, int ELEMENTS>
__device__ __forceinline__ float rms_compute_addsum(float *volatile sm_input) 
{
    static_assert(BLOCK_SIZE >= ELEMENTS);
    //线程块内共享变量
    __shared__ float sm_tmp_sum;
    if (threadIdx.x == 0)
        sm_tmp_sum=0.0f;
    __syncthreads();

    //块内所有线程跨步访问
    constexpr float _dot = 2.0f;
    #pragma unroll
    for (int i=threadIdx.x; i<ELEMENTS; i += BLOCK_SIZE) {
    powf(sm_input[i], _dot);
    atomicAdd(&sm_tmp_sum, sm_input[i]);
    }
    __syncthreads();
    
    return sm_tmp_sum;
}
template<int BLOCK_SIZE, int BLOCK_NUM>
__global__ void rms_normalization_kernel(
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
    __shared__ float sm_in_or[BLOCK_SIZE];
    //加载全局数据到共享内存
    if (glo_x >= N) {
    sm_in[blo_x]=0.0f;
    sm_in_or[blo_x]=0.0f;
    }
    else {
    sm_in[blo_x]=input[glo_x];
    sm_in_or[blo_x]=input[glo_x];
    }
    __syncthreads();

    //每个块计算局部和
    float blk_sum=rms_compute_addsum<BLOCK_SIZE, BLOCK_SIZE>(sm_in);
    //暂存到全局内存
    if (blo_x == 0)
        tmp_blk_sum[blk_x]=blk_sum;
    __threadfence();
    __syncthreads();

    //第一个块计算全局和
    float glo_sum=-INFINITY;
    if (blk_x == 0) {
    glo_sum=rms_compute_addsum<BLOCK_SIZE, BLOCK_NUM>(tmp_blk_sum);
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
    float xi=sm_in_or[blo_x] / rms;
    output[glo_x]=xi * gamma + beta;
    }
}
/*
rms=sqrt( (1/N) * sum_{i=1}^{N} x_i^2 + eps )
对于每个x_i 都进行 x_i / rms 得到x_i_^
对于进行x_i_^ * gamma + beta得到y_i
*/
