#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! 二维网格线程尺寸覆盖最长的边
template<int TIDSIZE_X, int TIDSIZE_Y>
__global__ void matrix_transpose_nature(
    const float* input, float* output, 
    int rows, int cols
) 
{
    int x_glo=blockIdx.x * TIDSIZE_X + threadIdx.x;
    int y_glo=blockIdx.y * TIDSIZE_Y + threadIdx.y;

    if (x_glo >= cols || y_glo >= rows)
    return ;
    float value=input[y_glo * cols + x_glo];
    //交换行列
    int index=y_glo + x_glo * rows;
    output[index]=value;
}
