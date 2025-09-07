#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 非对称式分块BM BK 控制内积维度BN
template<int BLOCK_X, int BLOCK_Y, int BN>
__global__ void matrix_multiplication_1(
    const float* A, const float* B, float* C, 
    int M, int N, int K, float alpha, float beta
)
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int glo_x=blockIdx.x * BLOCK_X + blo_x;
    int glo_y=blockIdx.y * BLOCK_Y + blo_y;
    //A加载BLOCK_Y行 BN个列片段元素
    __shared__ float As[BLOCK_Y][BN];
    //B加载BLOCK_X列 BN个行片段元素
    __shared__ float Bs[BN][BLOCK_X];

    float sum=0.0f;
    //K公共边 一次跨步加载BN 至少需要(K+BN-1)/BN跨步次数
    for (int n=0; n<(K + BN - 1) / BN; n++) {
    //一次写入BLOCK_Y行 BLOCK_X为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_x; i<BN; i=i + BLOCK_X) {
    int a_row=glo_y;
    int a_col=n * BN + i; //当前这次跨步对应到A全局列
    if (a_row < M && a_col < K)
    As[blo_y][i]=A[a_row * K + a_col];
    else
    As[blo_y][i]=0.0f;
    }

    //一次写入BLOCK_X列 BLOCK_Y为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_y; i<BN; i=i + BLOCK_Y) {
    int b_row=n * BN + i; //当前跨步对应B全局行
    int b_col=glo_x;
    if (b_row < K && b_col < N)
    Bs[i][b_col]=B[b_row * N + b_col];
    else
    Bs[i][b_col]=0.0f;
    }

    __syncthreads();

    float t_sum=0.0f;
    //每个线程内积运算
    #pragma unroll
    for (int i=0; i<BN; i++)
    t_sum+=As[blo_y][i] * Bs[i][blo_x];

    sum+=t_sum;
    __syncthreads();
    }

    if (glo_y >= M || glo_x >= N)
    return ;
    else {
    float old_c=C[glo_y * N + glo_x];
    C[glo_y * N + glo_x]=alpha * sum + old_c * beta;
    }
}
