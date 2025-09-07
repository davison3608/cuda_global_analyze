#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 二维网格线程尺寸覆盖M x N
template<int TIDSIZE_X, int TIDSIZE_Y>
__global__ void matrix_multiplication_nature(
    const float* A, const float* B, float* C, 
    int M, int N, int K
) 
{
    int glo_x=blockIdx.x * TIDSIZE_X + threadIdx.x;
    int glo_y=blockIdx.y * TIDSIZE_Y + threadIdx.y;
    int glo_idx=glo_x + glo_y * (gridDim.x * TIDSIZE_X);

    float sum=0;
    //内积运算
    if (glo_y >= M || glo_x >= N)
    return ;
    for (int d=0; d<K; d++) {
    //矩阵A对应一行
    float _a=A[glo_y * K + d]; 
    //矩阵B对应一列
    float _b=B[glo_x + d * N];
    sum+=_a * _b;
    }
    
    C[glo_y * N + glo_x]=sum;
}
