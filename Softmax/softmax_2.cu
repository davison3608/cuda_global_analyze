#include "../cudapacked.h"
//! 一维线程块 一维网格
//! 网格线程尺寸覆盖到N
//! warp归约 保证TIDNUM <= 32
template<int ELEMENTS>
__device__ __inline__ float softmax_reduce_max2(const float *volatile input)
{
    constexpr int WRSIZE = 32;
    constexpr int MASK = 0xFFFFFFFF;
    static_assert(ELEMENTS <= WRSIZE);
    int blo_x=threadIdx.x;

    float val=-INFINITY;
    if (blo_x < ELEMENTS)
        val=input[blo_x];
    __syncwarp(MASK);

    #pragma unroll
    for (int i=WRSIZE / 2; i>0; i=i / 2) {
    if (blo_x < i && (blo_x + i) < ELEMENTS)
    val=fmax(val, __shfl_xor_sync(MASK, val, i, WRSIZE)); 
    //无需__syncwarp
    }
    //0线程最值广播
    val=__shfl_sync(MASK, val, 0, WRSIZE);
    return val;   
}
template<int ELEMENTS>
__device__ __inline__ float softmax_reduce_sum2(const float*volatile input)
{
    constexpr int WRSIZE = 32;
    constexpr int MASK = 0xFFFFFFFF;
    static_assert(ELEMENTS <= WRSIZE);
    int blo_x=threadIdx.x;

    float sum=0.0f;
    if (blo_x < ELEMENTS)
        sum=input[blo_x];
    __syncwarp(MASK);

    for (int i=WRSIZE / 2; i>0; i=i / 2) {
    if (blo_x < i && (blo_x + i) < ELEMENTS)
    sum=sum + __shfl_xor_sync(MASK, sum, i, WRSIZE);
    //无需__syncwarp
    }
    //0线程广播
    sum=__shfl_sync(MASK, sum, 0, WRSIZE);
    return sum;
}
template<int TIDNUM>  
__global__ void softmax_kernel_2(
    const float* input, float* output, int N, 
    float* tmp_max, float* tmp_sum,            
    float* d_block_maxes, float* d_block_sums        
)
{
    constexpr int WRSIZE = 32;
    constexpr int MASK = 0xFFFFFFFF;
    static_assert(TIDNUM <= WRSIZE);
    int blo_x=threadIdx.x;
    int blk_x=blockIdx.x;
    int glo_x=blo_x + blk_x * WRSIZE;

    __shared__ float sm_in[WRSIZE];
    if (glo_x >= N)
    sm_in[blo_x]=-INFINITY;
    else
    sm_in[blo_x]=input[glo_x];
    __syncwarp(MASK);
    //每个块计算局部最值 由首线程写入一次
    float blk_max=softmax_reduce_max2<WRSIZE>(sm_in);
    if (blo_x == 0)
        d_block_maxes[blk_x]=blk_max;
    __threadfence();
    __syncwarp(MASK);

    //第一个块计算全局最值
    if (blk_x == 0) {
    float glo_max=softmax_reduce_max2<TIDNUM>(d_block_maxes);
    if (blo_x == 0)
        *tmp_max=glo_max;
    }
    __threadfence();
    __syncwarp(MASK);
    //广播到所有线程
    float glo_max=*tmp_max;
    __syncwarp(MASK);

    __shared__ float sm_exps[WRSIZE];
    //指数计算
    float exp_val=expf(sm_in[blo_x] - glo_max);
    sm_exps[blo_x]=exp_val; 
    __syncwarp(MASK);

    //每个块求局部和      
    float blk_sum=softmax_reduce_sum2<WRSIZE>(sm_exps);
    if (blo_x == 0)
        d_block_sums[blk_x]=blk_sum;
    __threadfence();
    __syncwarp(MASK);
    
    //第一个块计算全局总和
    if (blk_x == 0) {
    float glo_sum=softmax_reduce_sum2<TIDNUM>(d_block_sums);
    if (blo_x == 0)
        *tmp_sum=glo_sum;
    }
    __threadfence();
    __syncwarp(MASK);
    //广播到所有线程
    float glo_sum=*tmp_sum;
    __syncwarp(MASK);

    //softmax
    if (glo_x >= N)
    return ;
    else
    output[glo_x]=exp_val / glo_sum;
}
