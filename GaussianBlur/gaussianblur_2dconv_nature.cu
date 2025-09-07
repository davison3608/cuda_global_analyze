#include "../cudapacked.h"
//! 二维线程块 二维网格 单通道矩阵
//! 二维网格线程尺寸覆盖输入矩阵尺寸加上填充区域尺寸
template<int BLOCK_X, int BLOCK_Y>
__global__ void GaussianBlur_2d_base(
    const float* matrix, const float* kernel, float* output,
    int width, int height, 
    int ker_w, int ker_h
) 
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int glo_x=BLOCK_X * blockIdx.x + blo_x;
    int glo_y=BLOCK_Y * blockIdx.y + blo_y;
    //计算填充
    int padding_rows = (ker_h - 1) / 2;
    int padding_cols = (ker_w - 1) / 2;
    
    if (glo_x >= width || glo_y >= height)
    return ;

    float sum = 0.0f;
    //每个线程对应到输出矩阵一个元素
    for (int i=0; i<ker_h; i++) {
    for (int j=0; j<ker_w; j++) {
    int im_row=glo_y - padding_rows + i;  //邻域像素的行坐标
    int im_col=glo_x - padding_cols + j;  //邻域像素的列坐标
    //邻域边界检查
    if (im_row < 0 || im_row >= height || im_col < 0 || im_col >= width)
    continue ;
    float src_v=matrix[im_row * width + im_col];  
    float ker_v=kernel[i * ker_w + j];         
    //加权求和
    sum += src_v * ker_v;
    }
    }
    //写回
    output[glo_y * width + glo_x]=sum;
}
