#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 非对称式分块BM BK 控制内积维度BN
//! 单个线程计算更多元素 C子矩阵再次分块并保证对齐到2x2
//! 二维网格线程尺寸x2 覆盖C矩阵尺寸
template<int BLOCK_X, int BLOCK_Y, int BN, int THREADS = 2>
__global__ void matrix_multiplication_2(
    const float* A, const float* B, float* C, 
    int M, int N, int K, float alpha, float beta
)
{   
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    //每个线程在C的索引基准 对应2x2矩阵
    int col_base=blockIdx.x * BLOCK_X * THREADS + blo_x * THREADS;  
    int row_base=blockIdx.y * BLOCK_Y * THREADS + blo_y * THREADS;
    //共享内存适应2x2扩展
    //一次跨步处理A的BLOCK_Y * THREADS行 处理B的BLOCK_X * THREADS列
    __shared__ float As[BLOCK_Y * THREADS][BN];
    __shared__ float Bs[BN][BLOCK_X * THREADS];

    //2x2内积结果
    float sum[THREADS][THREADS] = {0.0f};

    //一次加载BN片段元素 至少需要(K+BN-1)/BN次跨步
    for (int n=0; n<(K + BN -1) / BN; n++) {
    #pragma unroll
    for (int i=blo_x; i<BN; i=i + BLOCK_X) {
    //以BLOCK_X为基准直到写入了BN个数量
    for (int dy=0; dy<THREADS; dy++) {
    //该线程访问A的THREADS行
    int a_row=row_base + dy;
    //这次跨步对应的A全局列
    int a_col=n * BN + i;
    //对应块内行
    int a_row_blo=blo_y * THREADS + dy;
    if (a_row < M && a_col < K)
    As[a_row_blo][i]=A[a_row * K + a_col];
    else
    As[a_row_blo][i]=0.0f;
    }
    }

    #pragma unroll
    for (int i=blo_y; i<BN; i=i + BLOCK_Y) {
    //以BLOCK_Y为基准直到写入了BN个数量
    for (int dx=0; dx<THREADS; dx++) {
    //这次跨步的对应B全局行
    int b_row=n * BN + i;
    //该线程访问B的THREADS列
    int b_col=col_base + dx;
    //对应块内列
    int b_col_blo=blo_y * THREADS + dy;
    if (b_row < K && b_col < N)
    Bs[i][b_col_blo]=B[b_row * N + b_col];
    else
    Bs[i][b_col_blo]=0.0f;
    }
    }

    __syncthreads();

    float _a[THREADS]={0.0f};
    float _b[THREADS]={0.0f};
    //每个线程内积
    #pragma unroll
    for (int i=0; i<BN; i++) {
    //一个线程取出As的THREADS行 取出Bs的THREADS列
    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    _a[dy]=As[blo_y * THREADS + dy][i];
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) 
    {_b[dx]=Bs[i][blo_x * THREADS + dx]; sum[dy][dx]+=_a[dy] * _b[dx];}
    }
    }
    __syncthreads();
    }

    //每个子矩阵所有元素线性变化
    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) {
    int glo_y=row_base + dy;
    int glo_x=col_base + dx;
    if (glo_y >= M || glo_x >= N)
    return ;
    else {
    float value=sum[dy][dx] * alpha + C[glo_y * N + glo_x] * beta;
    C[glo_y * N + glo_x]=value;
    }
    }
    }
}
