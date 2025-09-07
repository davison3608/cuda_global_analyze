#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! A B均行优先储存
//! 非对称式分块BM BK 控制内积维度BN
//! 单个线程计算更多元素 C子矩阵再次分块并保证对齐到2x2
//! 二维网格线程尺寸x2 覆盖C矩阵尺寸
//! BN对齐到float4 公共边K对齐到float4
template<int BLOCK_X, int BLOCK_Y, int BN, int THREADS = 2, int LEN = 4>
__global__ void matrix_multiplication_2(
    const float* A, const float* B, float* C, 
    int M, int N, int K, float alpha, float beta
)
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    //每个线程在C的2x2矩阵起始索引基准
    int col_base=blockIdx.x * BLOCK_X * THREADS + blo_x * THREADS;  
    int row_base=blockIdx.y * BLOCK_Y * THREADS + blo_y * THREADS;
    float4 *A_4=reinterpret_cast<float4*>(A);
    //向量化内积维度 向量化公共边
    const int BN_vec=BN / LEN;
    const int K_vec=K / LEN;
    
    //共享内存适应2x2扩展
    //一次跨步处理A的BLOCK_Y * THREADS行 处理B的BLOCK_X * THREADS列
    __shared__ float As[BLOCK_Y * THREADS][BN];
    __shared__ float Bs[BN][BLOCK_X * THREADS];

    //2x2内积结果
    float sum[THREADS][THREADS] = {0.0f};

    //一次加载BN片段元素 至少需要(K+BN-1)/BN次跨步
    for (int n=0; n<(K + BN - 1) / BN; n++) {
    #pragma unroll
    for (int i=blo_x; i<BN_vec; i=i + BLOCK_X) {
    //以BLOCK_X * 4为基准直到写入了BN个数量
    for (int dy=0; dy<THREADS; dy++) {
    //该线程访问A的THREADS行
    int a_row=row_base + dy;
    //这次跨步对应的A全局列 向量起始
    int a_col_base=n * BN + i;
    //对应块内行
    int a_row_blo=blo_y * THREADS + dy;
    if (a_row < M && a_col_base < K_vec) {
        float4 tmp=A_4[a_row * K_vec + a_col_base];
        As[a_row_blo][i + 0]=tmp.x;
        As[a_row_blo][i + 1]=tmp.y;
        As[a_row_blo][i + 2]=tmp.z;
        As[a_row_blo][i + 3]=tmp.w;
    }
    else {
        As[a_row_blo][i + 0]=0.0f;
        As[a_row_blo][i + 1]=0.0f;
        As[a_row_blo][i + 2]=0.0f;
        As[a_row_blo][i + 3]=0.0f;
    }
    }
    }

    #pragma unroll
    for (int i=blo_y; i<BN_vec; i=i + BLOCK_Y) {
    //以BLOCK_Y * 4为基准直到写入了BN个数量
    for (int dx=0; dx<THREADS; dx++) {
    //这次跨步的B全局行 向量起始
    int b_row_base=n * BN + i;
    //当前线程访问第THREADS列
    int b_col=col_base + dx;
    //该线程块内列
    int b_col_blo=blo_y * THREADS + dy;
    if (b_row_base < K_vec && b_col < N) {
        Bs[i + 0][b_col_blo]=B[(b_row_base + 0) * N + b_col];
        Bs[i + 1][b_col_blo]=B[(b_row_base + 1) * N + b_col];
        Bs[i + 2][b_col_blo]=B[(b_row_base + 2) * N + b_col];
        Bs[i + 3][b_col_blo]=B[(b_row_base + 3) * N + b_col];
    }
    else {
        Bs[i + 0][b_col_blo]=0.0f;
        Bs[i + 1][b_col_blo]=0.0f;
        Bs[i + 2][b_col_blo]=0.0f;
        Bs[i + 3][b_col_blo]=0.0f;
    }
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
    for (int dy=0; dy<THREADS; dy++) {

    for (int dx=0; dx<THREADS; dx++) {
    int glo_x=col_base + dx;
    int glo_y=row_base + dy;
    if (glo_y >= M || glo_x >= N)
    return ;
    else {
    float value=C[glo_y * N + glo_x] * beta + alpha * sum[dy][dx];
    C[glo_y * N + glo_x]=value;
    }
    }
    }
}
