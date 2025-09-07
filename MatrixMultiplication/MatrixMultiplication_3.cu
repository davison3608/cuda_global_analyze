#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 非对称式分块BM BK 控制内积维度BN
//! BN对齐到float4 公共边K对齐到float4
template<int BLOCK_X, int BLOCK_Y, int BN, int LEN = 4>
__global__ void matrix_multiplication_3(
    const float* A, const float* B, float* C, 
    int M, int N, int K, float alpha, float beta
)
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int glo_x=blockIdx.x * BLOCK_X + blo_x;
    int glo_y=blockIdx.y * BLOCK_Y + blo_y;
    float4 *A_4=reinterpret_cast<float4*>(A);
    //向量化内积维度 向量化公共边
    const int BN_vec=BN / LEN;
    const int K_vec=K / LEN;
    
    //A一次加载BLOCK_Y行 向量化加载到BN个列片段
    __shared__ float As[BLOCK_Y][BN];
    //B一次加载BLOCK_X列 0-BLOCK_Y每个加载四行到BN个行片段
    __shared__ float Bs[BN][BLOCK_X];

    float sum=0.0f; //每个线程负责的累加
    //K公共边 一次跨步加载BN 至少需要(K+BN-1)/BN跨步次数
    for (int n=0; n<(K + BN - 1) / BN; n++) {
    //一次写入BLOCK_Y行 BLOCK_X * 4为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_x; i<BN_vec; i=i + BLOCK_X) {
    int a_row=glo_y;
    int a_col_base=n * BN + i;
    if (a_row < M && a_col_base < K_vec) {
        float4 tmp=A_4[a_row * K_vec + a_col_base];
        As[blo_y][i + 0]=tmp.x;
        As[blo_y][i + 1]=tmp.y;
        As[blo_y][i + 2]=tmp.z;
        As[blo_y][i + 3]=tmp.w;
    }
    else {
        As[blo_y][i + 0]=0.0f;
        As[blo_y][i + 1]=0.0f;
        As[blo_y][i + 2]=0.0f;
        As[blo_y][i + 3]=0.0f;
    }
    }

    //一次写入BLOCK_X列 BLOCK_Y * 4为基数直到跨步加载了BN
    #pragma unroll
    for (int i=blo_y; i<BN_vec; i=i + BLOCK_Y) {
    int b_row_base=n * BN + i;
    int b_col=glo_x;
    if (b_row_base < K_vec && b_col < N) {
        Bs[i + 0][blo_x]=B[(b_row_base + 0) * N + b_col];
        Bs[i + 1][blo_x]=B[(b_row_base + 1) * N + b_col];
        Bs[i + 2][blo_x]=B[(b_row_base + 2) * N + b_col];
        Bs[i + 3][blo_x]=B[(b_row_base + 3) * N + b_col];
    }
    else {
        Bs[i + 0][blo_x]=0.0f;
        Bs[i + 1][blo_x]=0.0f;
        Bs[i + 2][blo_x]=0.0f;
        Bs[i + 3][blo_x]=0.0f;
    }
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
