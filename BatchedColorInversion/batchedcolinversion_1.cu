#include "../cudapacked.h"
//! 三维网格 二维线程块 
//! 线程块尺寸对齐到2x2 每个线程处理2x2子像素矩阵 对于列方向一个像素使用uchar3加载 所以保证图像3通道
//! 每个Z对应一个批次 每个批次的二维网格线程尺寸x2 覆盖C矩阵尺寸
template<int BLOCK_X, int BLOCK_Y, int BATCHS, int CHANNELS = 3, int THREADS = 2>
__global__ void batched_invert_0(
    unsigned char* image, 
    int width, int height
)
{
    static_assert(CHANNELS == 3);
    int batch=blockIdx.z;
    int blo_x=threadIdx.x;
    int blo_y=threadIdx.y;
    //当前批次内的块索引
    int ba_blk_x=blockIdx.x;
    int ba_blk_y=blockIdx.y;
    //当前批次内每个线程对应的2x2像素矩阵起始基准
    int ba_col_base=ba_blk_x * BLOCK_X * THREADS + threadIdx.x * THREADS;
    int ba_row_base=ba_blk_y * BLOCK_Y * THREADS + threadIdx.y * THREADS;

    uchar3 *image3=reinterpret_cast<uchar3*>(image);
    //缓存像素
    __shared__ u_char sm_pixes[BLOCK_Y * THREADS][BLOCK_X * THREADS * CHANNELS];
    //每个线程加载2x2像素矩阵
    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) {
    //当前z对应批次的偏移 针对uchar3地址
    int ba_offset3=batch * width * height;
    //当前线程处理当前批次的确切像素起始基准
    int ba_col=ba_col_base + dx;
    int ba_row=ba_row_base + dy;
    //块内起始基准
    int blo_col=blo_x * THREADS + dx;
    int blo_row=blo_y * THREADS + dy;
    //加载一个uchar3像素
    uchar3 pix_3=make_uchar3(0, 0, 0);
    if (batch < BATCHS && ba_row < height && ba_col < width)
        pix_3=image3[ba_offset3 + ba_row * width + ba_col];
    sm_pixes[blo_row][blo_col * CHANNELS + 0]=pix_3.x;
    sm_pixes[blo_row][blo_col * CHANNELS + 1]=pix_3.y;
    sm_pixes[blo_row][blo_col * CHANNELS + 2]=pix_3.z;
    }
    }
    __syncthreads();

    #pragma unroll
    for (int dy=0; dy<THREADS; dy++) {
    #pragma unroll
    for (int dx=0; dx<THREADS; dx++) {
    //当前z对应批次的偏移 针对uchar3地址
    int ba_offset3=batch * width * height;
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
    //写回一个uchar3
    uchar3 pix_3=make_uchar3(
    sm_pixes[blo_row][blo_col * CHANNELS + 0],
    sm_pixes[blo_row][blo_col * CHANNELS + 1],
    sm_pixes[blo_row][blo_col * CHANNELS + 2]
    );
    image3[ba_offset3 + ba_row * width + ba_col]=pix_3;    
    }
    }
}
