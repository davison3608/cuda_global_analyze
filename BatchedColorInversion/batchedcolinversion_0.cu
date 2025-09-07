#include "../cudapacked.h"
//! 三维网格 二维线程块
//! 线程块尺寸对齐到2x2 每个线程处理2x2子像素矩阵
//! 每个Z对应一个批次 每个批次的二维网格线程尺寸x2 覆盖C矩阵尺寸
template<int BLOCK_X, int BLOCK_Y, int BATCHS, int CHANNELS, int THREADS = 2>
__global__ void batched_invert_0(
    unsigned char* image, 
    int width, int height
)
{
    int batch=blockIdx.z;
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    //当前批次内的块索引
    int ba_blk_x=blockIdx.x;
    int ba_blk_y=blockIdx.y;
    //当前批次内每个线程对应的2x2像素矩阵起始基准
    int ba_col_base=ba_blk_x * BLOCK_X * THREADS + threadIdx.x * THREADS;
    int ba_row_base=ba_blk_y * BLOCK_Y * THREADS + threadIdx.y * THREADS;

    //缓存像素
    __shared__ u_char sm_pixes[BLOCK_Y * THREADS][BLOCK_X * THREADS * CHANNELS];
    //每个线程加载2x2子矩阵像素
    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) {
    //当前z对应批次的偏移
    int ba_offset=batch * width * height * CHANNELS;
    //当前线程处理当前批次的确切像素起始基准
    int ba_col=ba_col_base + dx;
    int ba_row=ba_row_base + dy;
    //块内起始基准
    int blo_col=blo_x * THREADS + dx;
    int blo_row=blo_y * THREADS + dy;
    //加载CHANNELS个uchar
    #pragma unroll
    for (int c=0; c<CHANNELS; c++) {
    //全局内存地址 批次偏移 + 行偏移 + 列偏移 + 通道
    int global_uchar_idx=ba_offset + ba_row * width * CHANNELS + ba_col * CHANNELS + c;
    if (batch < BATCHS && ba_row < height && ba_col < width)
        sm_pixes[blo_row][blo_col * CHANNELS + c]=image[global_uchar_idx];
    else
        sm_pixes[blo_row][blo_col * CHANNELS + c]=0;
    }
    }
    }
    __syncthreads();

    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) {
    //当前z对应批次的偏移
    int ba_offset=batch * width * height * CHANNELS;
    //当前线程处理当前批次的确切像素起始基准
    int ba_col=ba_col_base + dx;
    int ba_row=ba_row_base + dy;
    //跳过不符合边界像素
    if (batch >= BATCHS || ba_row >= height || ba_col >= width)
        continue ;
    //块内起始基准
    int blo_col=blo_x * THREADS + dx;
    int blo_row=blo_y * THREADS + dy;
    //反转
    sm_pixes[blo_row][blo_col * CHANNELS + 0]=255 - sm_pixes[blo_row][blo_col * CHANNELS + 0];
    sm_pixes[blo_row][blo_col * CHANNELS + 1]=255 - sm_pixes[blo_row][blo_col * CHANNELS + 1];
    sm_pixes[blo_row][blo_col * CHANNELS + 2]=255 - sm_pixes[blo_row][blo_col * CHANNELS + 2];
    //写回
    int global_uchar_idx_base=ba_offset + ba_row * width * CHANNELS + ba_col * CHANNELS;
    image[global_uchar_idx_base + 0]=sm_pixes[blo_row][blo_col * CHANNELS + 0];
    image[global_uchar_idx_base + 1]=sm_pixes[blo_row][blo_col * CHANNELS + 1];
    image[global_uchar_idx_base + 2]=sm_pixes[blo_row][blo_col * CHANNELS + 2];
    }
    }
}
