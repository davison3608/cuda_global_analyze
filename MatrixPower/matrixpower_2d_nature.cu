#include "../cudapacked.h"
//! 二维网格 二维线程块
//! 二维网格线程尺寸覆盖到NxN
template<int TIDSIZE, int N>
__global__ void matrix_power_n_2d_nature(
    float *A, int power,
    float *tmp_mat
)
{
    int glo_x=blockIdx.x * TIDSIZE + threadIdx.x;
    int glo_y=blockIdx.y * TIDSIZE + threadIdx.y;
    //边界处理
    if (glo_x >= N || glo_y >= N)
    return ;
    //对于0次幂 单位矩阵
    if (power == 0) 
    tmp_mat[glo_y * N + glo_x]=(glo_x == glo_y) ? 1.0f : 0.0f;
    //对于1次幂 原矩阵
    else if (power == 1)
    tmp_mat[glo_y * N + glo_x]=A[glo_y * N + glo_x];
    
    constexpr int C_M = N;
    constexpr int C_N = N;
    constexpr int C_K = N;
    //功率计算 第一次乘法
    float sum = 0.0f;
    for (int k = 0; k < C_K; k++) {
    float _a = A[glo_y * C_K + k];  
    float _b = A[glo_x + k * C_N];  
    sum += _a * _b;
    }
    tmp_mat[glo_y * N + glo_x] = sum;
    //确保第一次乘法完成
    __syncthreads();
    
    //剩余power-2次
    for (int p = 2; p < power; p++) {
    sum = 0.0f;
    for (int k = 0; k < N; k++) {
    float _a = tmp_mat[glo_y * C_K + k];  // 使用上一次的结果
    float _b = A[glo_x + k * C_N];
    sum += _a * _b;
    }
    float tmp_sum=sum;
    //存在临时变量 避免获取到最新值
    __syncthreads();
    tmp_mat[glo_y * C_N + glo_x]=tmp_sum;
    //确保每一次完成
    __syncthreads(); 
    }

    //写回
    A[glo_y * C_N + glo_x]=tmp_mat[glo_y * C_N + glo_x];
}
