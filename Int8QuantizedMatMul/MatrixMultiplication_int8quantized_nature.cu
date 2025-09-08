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
template<int BLOCK_X, int BLOCK_Y>
__global__ void quantized_matmul_kernel(
    const int8_t* A, const int8_t* B, int8_t* C, 
    int M, int N, int K, 
    float scale_A, float scale_B, float scale_C, 
    int zero_point_A, int zero_point_B, int zero_point_C
) 
{
    int glo_x=blockIdx.x * BLOCK_X + threadIdx.x;
    int glo_y=blockIdx.y * BLOCK_Y + threadIdx.y;
    if(glo_x >= N || glo_y >= M)
    return ; 
    int sum = 0.0;
    for(int k = 0; k < K; ++k) 
    sum +=(A[glo_y*K+k] - zero_point_A) * (B[k*N+glo_x] - zero_point_B);
    
    C[glo_y*N+glo_x]=clamp(lroundf(sum*scale_B*scale_A/scale_C) + zero_point_C, -128, 127);
}
