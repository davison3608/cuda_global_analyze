#include "../cudapacked.h"
//! 一维度线程块 一维网格
//! 网格线程尺寸覆盖到N
//! 原子计算最大值与求和
__device__ __forceinline__ float atomicMax(float* address, float value) {
    //float二进制表示转换为unsigned int
    unsigned int* addr_as_uint = (unsigned int*)address;
    unsigned int old = *addr_as_uint;
    unsigned int val_as_uint = *(unsigned int*)&value;

    //循环 如果当前值小于value则更新
    while (val_as_uint > old) {
        old = atomicCAS(addr_as_uint, old, val_as_uint);
        val_as_uint = *(unsigned int*)&value;  
    }
    return *(float*)&old;
}
template<int TIDSIZE, int ELEMENTS>
__device__ float softmax_atomic_max(const float* volatile input) 
{
    static_assert(ELEMENTS <= TIDSIZE);
    __shared__ float shared_max;  

    if (threadIdx.x == 0) {
        shared_max = input[0];
    }
    __syncthreads(); 

    //所有线程跨步遍历元素
    #pragma unroll  
    for (int i = threadIdx.x; i < ELEMENTS; i += TIDSIZE) {
        atomicMax(&shared_max, input[i]);  
    }
    __syncthreads();  

    return shared_max;
}
template<int TIDSIZE, int ELEMENTS>
__device__ __inline__ int softmax_atomic_sum(const float* volatile input) 
{
    static_assert(ELEMENTS <= TIDSIZE);
    __shared__ float shared_sum;  

    if (threadIdx.x == 0) 
    shared_sum = 0.0f;
    __syncthreads();  

    //所有线程跨步遍历元素
    #pragma unroll
    for (int i = threadIdx.x; i < ELEMENTS; i += TIDSIZE) 
    atomicAdd(&shared_sum, input[i]);  
    __syncthreads();  
    
    return shared_sum;
}
template<int TIDSIZE, int TIDNUM>  
__global__ void softmax_kernel_0(
    const float* input, float* output, int N, 
    float* tmp_max, float* tmp_sum,            
    float* d_block_maxes, float* d_block_sums        
) 
{
    static_assert(TIDNUM <= TIDSIZE);
    const int blo_x = threadIdx.x;  
    const int blk_x = blockIdx.x;   
    const int glo_x = blk_x * TIDSIZE + blo_x;  // 全局线程索引

    __shared__ float sm_in[TIDSIZE];
    //加载输入数据
    if (glo_x >= N) 
    sm_in[blo_x] = -INFINITY;
    else 
    sm_in[blo_x] = input[glo_x];
    __syncthreads(); 

    //每个块计算最大值 写入全局
    const float blk_max = softmax_atomic_max<TIDSIZE, TIDSIZE>(sm_in);
    //每个块线程0写入
    if (blo_x == 0) 
        d_block_maxes[blk_x] = blk_max;
    __threadfence();
    __syncthreads();

    //第一个块计算全局最大
    float glo_max = -INFINITY;
    if (blk_x == 0) {  
    glo_max = softmax_atomic_max<TIDSIZE, TIDNUM>(d_block_maxes);
    //写入全局广播给所有块
    if (blo_x == 0) 
        *tmp_max = glo_max;      
    }
    __threadfence();  
    __syncthreads();

    glo_max = *tmp_max;
    __syncthreads();

    __shared__ float sm_exps[TIDSIZE];  
    //每个块计算指数 
    //无需边界处理 exp负无穷为0
    float exp_val = expf(sm_in[blo_x] - glo_max); 
    sm_exps[blo_x] = exp_val;
    __syncthreads();

    //每个块指数总和 写入全局内存
    float blk_sum = softmax_atomic_sum<TIDSIZE, TIDSIZE>(sm_exps);
    if (blo_x == 0) 
        d_block_sums[blk_x] = blk_sum;
    __threadfence();  
    __syncthreads();

    //第一个块计算全局总和
    float glo_sum = 0.0f; 
    if (blk_x == 0) {  
    glo_sum = softmax_atomic_sum<TIDSIZE, TIDNUM>(d_block_sums);
    //写入全局广播到所有块
    if (blo_x == 0) 
        *tmp_sum = glo_sum;  
    }
    __threadfence();  
    __syncthreads();

    glo_sum = *tmp_sum;
    __syncthreads();
    //softmax
    if (glo_x < N) 
    output[glo_x] = exp_val / glo_sum;
    else
    return ;
}
/*
假设输入是一个 K 维向量 
1 求向量最值
2 所有值减去最值求指数
3 求和作分母
4 每个指数值除以总和得到概率向量
*/