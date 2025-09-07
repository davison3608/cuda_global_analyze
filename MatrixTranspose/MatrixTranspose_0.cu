#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! 二维网格线程尺寸覆盖最长的边
template<int TIDSIZE_X, int TIDSIZE_Y>
__global__ void matrix_transpose_0(
    const float* __restrict__ input, float* __restrict__ output, 
    int rows, int cols
) 
{
    int x_glo=blockIdx.x * TIDSIZE_X + threadIdx.x;
    int y_glo=blockIdx.y * TIDSIZE_Y + threadIdx.y;

    __shared__ float smem[TIDSIZE_Y][TIDSIZE_X];

    if (x_glo >= rows || y_glo >= cols)
    return ;
    smem[threadIdx.y][threadIdx.x]=input[y_glo * cols + x_glo];
    __syncthreads();

    //交换行列
    int tran_row=x_glo;
    int tran_col=y_glo;
    int index=tran_col + tran_row * rows;
    //这个元素在share内交换行列
    output[index]=smem[threadIdx.x][threadIdx.y];
}