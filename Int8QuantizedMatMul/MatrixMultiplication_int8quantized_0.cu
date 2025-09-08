#include "../cudapacked.h"
//! 二维线程块 二维网格
//! 二维网格线程尺寸覆盖结果矩阵尺寸
__device__ long clamp(long x, long a, long b) {
    if(x < a) 
        return a;
    else if(b < x) 
        return b;
    else 
        return x;
}
template<int BLOCK_X, int BLOCK_Y, int BN>
__global__ void quantized_matmul_kernel_0(
    const int8_t* A, const int8_t* B, int8_t* C, 
    int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, 
    int zero_p_A, int zero_p_B, int zero_p_C
)
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int blk_x=blockIdx.x;
    int blk_y=blockIdx.y;
    int glo_x = blk_x * BLOCK_X + blo_x;
    int glo_y = blk_y * BLOCK_Y + blo_y;

    __shared__ int8_t sm_As[BLOCK_Y][BLOCK_X];
    __shared__ int8_t sm_Bs[BLOCK_Y][BLOCK_X];
    int sum = 0;
    //K公共边 一次跨步加载BN 至少需要(K+BN-1)/BN跨步次数
    for(int stride=0; stride<(K + BN - 1)/BN; stride++) {
    //一次写入BLOCK_Y行 BLOCK_X为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_x; i<BN; i=i + BLOCK_X) {
    int a_row=glo_y;
    int a_col=stride * BN + i; //当前这次跨步对应到A全局列
    if(a_row < M && a_col < K)
    sm_As[blo_y][i]=A[a_row * K + a_col];
    else
    sm_As[blo_y][i]=0;
    }
    
    //一次写入BLOCK_X列 BLOCK_Y为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_y; i<BN; i=i + BLOCK_Y) {
    int b_row=stride * BN + i; //当前跨步对应B全局行
    int b_col=glo_x;
    if(b_row < K && b_col < N)
    sm_Bs[i][blo_x]=B[b_col + b_row * N];
    else
    sm_Bs[i][blo_x]=0;
    }
    
    __syncthreads();

    //每个线程内积运算
    #pragma unroll
    for(unsigned int i = 0; i < BN; i++)
    sum += (sm_As[blo_y][i] - zero_p_A) * (sm_Bs[i][blo_x] - zero_p_B);
    __syncthreads();
    }

    if(glo_y >= M || glo_x >= N)
    return ;
    C[glo_y * N + glo_x]=clamp(lroundf(sum * scale_A * scale_B / scale_C) + zero_p_C, -128, 127);
}