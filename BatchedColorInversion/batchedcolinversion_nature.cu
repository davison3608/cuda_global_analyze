#include "../cudapacked.h"
//! 三维网格 二维线程块 
//! 每个Z对应一个批次 每个批次二维网格线程尺寸覆盖图像尺寸
template<int BLOCK_X, int BLOCK_Y, int BATCHS, int CHANNELS>
__global__ void batched_invert_kernel(
    unsigned char* image, 
    int width, int height
) 
{
    int batch=blockIdx.z;
    //当前批次内的块索引
    int ba_blk_x=blockIdx.x;
    int ba_blk_y=blockIdx.y;
    //当前批次内的全局线程索引
    int ba_glo_x=ba_blk_x * BLOCK_X + threadIdx.x;
    int ba_glo_y=ba_blk_y * BLOCK_Y + threadIdx.y;

    if (batch >= BATCHS || ba_glo_x >= width || ba_glo_y >= height)
    return ;
    else {
    //当前z对应批次
    int ba_offset=batch * width * height * CHANNELS;
    //当前批次像素起始
    int ba_pix_offset=ba_glo_y * width + ba_glo_x;
    u_char r=image[ba_offset + (ba_pix_offset * CHANNELS) + 0];
    u_char g=image[ba_offset + (ba_pix_offset * CHANNELS) + 1];
    u_char b=image[ba_offset + (ba_pix_offset * CHANNELS) + 2];
    image[ba_offset + (ba_pix_offset * CHANNELS) + 0]=255 - r;
    image[ba_offset + (ba_pix_offset * CHANNELS) + 1]=255 - g;
    image[ba_offset + (ba_pix_offset * CHANNELS) + 2]=255 - b;
    }
}
//! 图像必须为跨步格式