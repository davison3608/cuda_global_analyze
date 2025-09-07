#include "../cudapacked.h"
//! 二维线程块 二维网格 单通道矩阵
//! 线程块尺寸覆盖权重核尺寸
//! 二维网格线程尺寸覆盖输入矩阵尺寸加上填充区域尺寸
template<int BLOCK_X, int BLOCK_Y, int KERNEL_X, int KERNEL_Y>
__global__ void GaussianBlur_2d_0(
    const float* matrix, const float* kernel, float* output,
    int width, int height
) 
{
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    int glo_x=BLOCK_X * blockIdx.x + blo_x;
    int glo_y=BLOCK_Y * blockIdx.y + blo_y;
    //计算填充
    constexpr int PAD_ROW=(KERNEL_Y - 1) / 2;
    constexpr int PAD_COL=(KERNEL_X - 1) / 2;
    //缓存输入附加填充 对于位于中间的块则仍然加载的是邻近区域数据
    __shared__ float sm_in[BLOCK_Y + PAD_ROW * 2][BLOCK_X + PAD_COL * 2];
    //先加载共享内存的有效区域
    int sm_x=blo_x + PAD_COL; //PAD_COL ~ PAD_COL + BLOCK_X - 1
    int sm_y=blo_y + PAD_ROW; //PAD_ROW ~ PAD_ROW + BLOCK_Y - 1
    //计算该输出像素所需的输入邻域区域的起始坐标
    int im_x=glo_x - PAD_COL;
    int im_y=glo_y - PAD_ROW;
    if (im_x < width && im_x >= 0 && im_y < height && im_y >= 0)
    sm_in[sm_y][sm_x]=matrix[im_y * width + im_x];
    else
    sm_in[sm_y][sm_x]=0.0f; //越界则邻近区域置0
    
    //左方向填充区域
    if (blo_x < PAD_COL) {
    int sm_x_pad=blo_x;
    int im_x_pad=im_x - PAD_COL; //有效区左移
    if (im_x_pad >= 0 && im_x_pad < width && im_y >= 0 && im_y < height)
    sm_in[sm_y][sm_x_pad]=matrix[im_y * width + im_x_pad];
    else
    sm_in[sm_y][sm_x_pad]=0.0f;
    }
    //右方向填充区域
    if (blo_x >= BLOCK_X - PAD_COL) {
    int sm_x_pad=blo_x + 2 * PAD_COL;
    int im_x_pad=im_x + PAD_COL; //有效区右移
    if (im_x_pad >= 0 && im_x_pad < width && im_y >= 0 && im_y < height)
    sm_in[sm_y][sm_x_pad]=matrix[im_y * width + im_x_pad];
    else
    sm_in[sm_y][sm_x_pad]=0.0f;
    }
    //上方向填充区域
    if (blo_y < PAD_ROW) {
    int sm_y_pad=blo_y;
    int im_y_pad=im_y - PAD_ROW; //有效区上移
    if (im_y_pad >= 0 && im_y_pad < height && im_x >= 0 && im_x < width)
    sm_in[sm_y_pad][sm_x]=matrix[im_y_pad * width + im_x];
    else
    sm_in[sm_y_pad][sm_x]=0.0f;
    }
    //下方向填充区域
    if (blo_y >= BLOCK_Y - PAD_ROW) {
    int sm_y_pad=blo_y + 2 * PAD_ROW;
    int im_y_pad=im_y + PAD_ROW; //有效区下移
    if (im_y_pad >= 0 && im_y_pad < height && im_x >= 0 && im_x < width)
    sm_in[sm_y_pad][sm_x]=matrix[im_y_pad * width + im_x];
    else
    sm_in[sm_y_pad][sm_x]=0.0f;
    }
    __syncthreads();

    __shared__ float sm_ker[KERNEL_Y][KERNEL_X];
    if (blo_x < KERNEL_X && blo_y < KERNEL_Y)
        sm_ker[blo_y][blo_x]=kernel[blo_y * KERNEL_X + blo_x];
    __syncthreads();

    float sum=0.0f;
    //核卷积阶段
    #pragma unroll
    for (int dy=0; dy<KERNEL_Y; dy++) {
    #pragma unroll
    for (int dx=0; dx<KERNEL_X; dx++)     
    sum+=sm_ker[dy][dx] * sm_in[blo_y + dy][blo_x + dx];
    }
    //写回
    if (glo_x >= width || glo_y >= height)
    return ;
    output[glo_y * width + glo_x]=sum;
}
