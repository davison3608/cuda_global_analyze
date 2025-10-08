#include "../cudapacked.h"
//! 一维线程块 一维网格
//! 向量N对齐到float4 网格线程尺寸覆盖到N / 4 
//! warp归约 保证WR_NUM <= 32
template<int ELEMENTS_LEN, int LEN = 4>
__device__ __forceinline__ float rms_compute_addsum_red_sf_vec(float *volatile sm_input) 
{
    constexpr int   WRSIZE = 32;
    constexpr int   MASK   = 0xFFFFFFFF;
    constexpr float _dot   = 2.0f;
    static_assert(ELEMENTS_LEN <= WRSIZE);
    //warp一个线程持有float4
    int blo_x=threadIdx.x;
    int sm_x_start= blo_x * LEN;

    float sum_thread=0.0f;
    if (blo_x < ELEMENTS_LEN) {
    sum_thread+=powf(sm_input[sm_x_start + 0], _dot);
    sum_thread+=powf(sm_input[sm_x_start + 1], _dot);
    sum_thread+=powf(sm_input[sm_x_start + 2], _dot);
    sum_thread+=powf(sm_input[sm_x_start + 3], _dot);
    }
    __syncwarp(MASK);

    #pragma unroll
    for (int i=WRSIZE / 2; i>0; i=i / 2) {
    if (blo_x < i && (blo_x + i) < ELEMENTS_LEN)
    sum_thread += __shfl_xor_sync(MASK, sum_thread, i, WRSIZE);
    //无需__syncwarp
    }
    //0线程广播
    sum_thread=__shfl_sync(MASK, sum_thread, 0, WRSIZE);
    return sum_thread;
}
template<int WR_NUM, int LEN = 4>
__global__ void rms_normalization_kernel_3(
    float* __restrict__ input, float gamma, float beta, 
    float* __restrict__ output, int N, float eps,
    float *tmp_sum, float *tmp_blk_sum
) 
{
    constexpr int WRSIZE = 32;
    static_assert(WR_NUM <= WRSIZE);
    const int blk_x = blockIdx.x;   
    const int blo_x = threadIdx.x;  
    const int glo_x = blk_x * WRSIZE + blo_x;  

    float4 *input4=reinterpret_cast<float4*>(input);
    float4 *output4=reinterpret_cast<float4*>(output);
    __shared__ float sm_in[WRSIZE * LEN];
    //块内线程映射到共享内存索引
    int sm_x_offset=blo_x * LEN;

    //加载全局数据到共享内存
    if (glo_x >= (N / LEN)) {
    sm_in[sm_x_offset + 0]=0.0f;
    sm_in[sm_x_offset + 1]=0.0f;
    sm_in[sm_x_offset + 2]=0.0f;
    sm_in[sm_x_offset + 3]=0.0f;
    } 
    else {
    float4 src4=input4[glo_x];
    sm_in[sm_x_offset + 0]=src4.x;
    sm_in[sm_x_offset + 1]=src4.y;
    sm_in[sm_x_offset + 2]=src4.z;
    sm_in[sm_x_offset + 3]=src4.w;
    }
    __syncthreads();

    //每个块计算局部和
    float blk_sum=rms_compute_addsum_red_sf_vec<WRSIZE, LEN>(sm_in);
    //暂存到全局内存
    if (blo_x == 0)
        tmp_blk_sum[blk_x]=blk_sum;
    __threadfence();
    __syncthreads();

    //第一个块计算全局和
    float glo_sum = 0.0f;
    if (blk_x == 0) {
    //tmp_blk_sum的前WR_NUM个元素 LEN指定为1
    glo_sum = rms_compute_addsum_red_sf_vec<WR_NUM, 1>(tmp_blk_sum);  
    // lane0线程将全局和写入输出变量tmp_sum
    if (blo_x == 0) 
        *tmp_sum = glo_sum;
    }
    __threadfence();  
    __syncthreads();

    glo_sum = *tmp_sum;
    __syncthreads();

    //计算rms变量
    float rms = sqrtf((glo_sum / N) + eps);

    //归一化后写回
    if (glo_x >= (N / LEN)) 
        return;
    else {
    float4 val4{0, 0, 0, 0};
    val4.x=sm_in[sm_x_offset + 0] / rms;
    val4.y=sm_in[sm_x_offset + 1] / rms;
    val4.z=sm_in[sm_x_offset + 2] / rms;
    val4.w=sm_in[sm_x_offset + 3] / rms;
    //gamma缩放与beta偏移
    val4.x=val4.x * gamma + beta;
    val4.y=val4.y * gamma + beta;
    val4.z=val4.z * gamma + beta;
    val4.w=val4.w * gamma + beta;
    output4[glo_x]=val4;
    }
}