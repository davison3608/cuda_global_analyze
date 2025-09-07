#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 二维网格线程尺寸覆盖M x N
template<int TIDSIZE>
__global__ void matrix_multiplication_0(
    const float* A, const float* B, float* C, 
    int M, int N, int K
)
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int glo_x=blockIdx.x * TIDSIZE + threadIdx.x;
    int glo_y=blockIdx.y * TIDSIZE + threadIdx.y;
    //As为TIDSIZE行片段 Bs为TIDSIZE列片段
    __shared__ float As[TIDSIZE][TIDSIZE];
    __shared__ float Bs[TIDSIZE][TIDSIZE];
    
    float sum=0.0f;
    //跨步加载 每次处理TIDSIZE
    for (int n=0; n<(K + TIDSIZE - 1) / TIDSIZE; n++) {
    int a_row=glo_y;
    int a_col=n * TIDSIZE + blo_x;
    int b_row=n * TIDSIZE + blo_y;
    int b_col=glo_x;
    //块所有线程取出A的TIDSIZE行片段 B的TIDSIZE列片段
    if (a_row < M && a_col < K)
    As[blo_y][blo_x]=A[a_row * K + a_col];
    else
    As[blo_y][blo_x]=0.0f;
    if (b_row < K && b_col < N)
    Bs[blo_y][blo_x]=B[b_row * N + b_col];
    else
    Bs[blo_y][blo_x]=0.0f;
    __syncthreads();

    float t_sum=0.0f;
    //每个线程片段内积
    #pragma unroll
    for (int i=0; i<TIDSIZE; i++)
    t_sum+=As[blo_y][i] * Bs[i][blo_x];
    
    sum+=t_sum;
    __syncthreads();
    }

    if (glo_y >= M || glo_x >= N)
    return ;
    else
    C[glo_y * N + glo_x]=sum;
}

//TIDSIZE_X TIDSIZE_Y对应的C子矩阵 
//A跨步加载TIDSIZE_Y行 每行TIDSIZE_X列片段元素 因为TIDSIZE_X属于同一行
//B跨步加载TIDSIZE_X列 每列TIDSIZE_Y行片段元素 因为TIDSIZE_Y属于同一列
