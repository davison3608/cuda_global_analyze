#include "../cudapacked.h"
//! 一维度线程块 一维网格
//! 网格线程尺寸覆盖到N
//! 归约计算最大值与求和
//! TIDSIZE保证>=TIDNUM保证块归约
template<int TIDSIZE, int ELEMENTS>
__device__ __inline__ float softmax_reduce_max_1(const float *volatile input)
{
    static_assert(ELEMENTS <= TIDSIZE);
    int blo_x=threadIdx.x;
    __shared__ float sm_maxs[ELEMENTS];

    if (blo_x < ELEMENTS) 
    sm_maxs[blo_x]=input[blo_x];
    else
    sm_maxs[blo_x]=-INFINITY;
    __syncthreads();

    #pragma unroll
    for (int n=TIDSIZE / 2; n>0; n=n / 2) {
    //每次n的跨步
    if (blo_x < n && (blo_x + n) < ELEMENTS)
    sm_maxs[blo_x]=fmax(sm_maxs[blo_x], sm_maxs[blo_x + n]); 
    //确保下次迭代同步
    __syncthreads();
    }

    return sm_maxs[0];
}
template<int TIDSIZE, int ELEMENTS>
__device__ __inline__ float softmax_reduce_sum_1(const float *volatile input)
{
    static_assert(ELEMENTS <= TIDSIZE);
    int blo_x=threadIdx.x;
    __shared__ float sm_sums[ELEMENTS];

    if (blo_x < ELEMENTS)
    sm_sums[blo_x]=input[blo_x];
    else
    sm_sums[blo_x]=-INFINITY;
    __syncthreads();

    #pragma unroll
    for (int n=TIDSIZE / 2; n>0; n=n / 2) {
    //每次n跨步
    if (blo_x < n && (blo_x + n) < ELEMENTS)
    sm_sums[blo_x]=sm_sums[blo_x] + sm_sums[blo_x + n];
    //确保下次迭代同步
    __syncthreads();
    }
    
    return sm_sums[0];
}
template<int TIDSIZE, int TIDNUM>  
__global__ void softmax_kernel_1(
    const float* input, float* output, int N, 
    float* tmp_max, float* tmp_sum,            
    float* d_block_maxes, float* d_block_sums        
)
{
    static_assert(TIDNUM <= TIDSIZE);
    int blo_x=threadIdx.x;
    int blk_x=blockIdx.x;
    int glo_x=blk_x * TIDSIZE + blo_x;

    __shared__ float sm_in[TIDSIZE];
    //加载输入数据
    if (glo_x >= N)
    sm_in[blo_x]=-INFINITY;
    else
    sm_in[blo_x]=input[glo_x];
    __syncthreads();

    //每个块归约求最值
    float blk_max=softmax_reduce_max_1<TIDSIZE, TIDSIZE>(sm_in);
    if (blo_x == 0) 
        d_block_maxes[blk_x]=blk_max;
    __threadfence();
    __syncthreads();

    //第一个块计算全局总和
    if (blk_x == 0) {
    float glo_max=softmax_reduce_max_1<TIDSIZE, TIDNUM>(d_block_maxes);
    if (blo_x == 0)
        *tmp_max=glo_max;
    }
    __threadfence();
    __syncthreads();
    //广播到所有块
    float glo_max=*tmp_max;
    __syncthreads();

    __shared__ float sm_exps[TIDSIZE];
    //指数计算
    float exp_val=expf(sm_in[blo_x] - glo_max);
    sm_exps[blo_x]=exp_val;
    __syncthreads();

    //每个块归约求和
    float blk_sum=softmax_reduce_sum_1<TIDSIZE, TIDSIZE>(sm_exps);
    if (blo_x == 0)
        d_block_sums[blk_x]=blk_sum;
    __threadfence();
    __syncthreads();

    //第一个块计算全局归约和
    if (blk_x == 0) {
    float glo_sum=softmax_reduce_sum_1<TIDSIZE, TIDNUM>(d_block_sums);
    if (blo_x == 0)
        *tmp_sum=glo_sum;
    }
    __threadfence();
    __syncthreads();
    //广播到所有块
    float glo_sum=*tmp_sum;
    __syncthreads();

    //softmax
    if (glo_x >= N)
    return ;
    else
    output[glo_x]=exp_val / glo_sum;
}
