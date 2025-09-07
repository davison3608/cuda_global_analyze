#include "../cudapacked.h"
//! 二维线程块 二维网格 
//! 二维网格线程尺寸x轴x4覆盖cols 保证cols对齐4倍数
//! y轴原生覆盖rows
template<int TIDSIZE_X, int TIDSIZE_Y>
__global__ void matrix_transpose_1(
    const float* __restrict__ input, float* __restrict__ output, 
    int rows, int cols
)
{
    int x4_glo=blockIdx.x * TIDSIZE_X + threadIdx.x;
    int y_glo=blockIdx.y * TIDSIZE_Y + threadIdx.y;

    float4 *input4=static_cast<float4*>(input);

    __shared__ float smem[TIDSIZE_Y][TIDSIZE_X * 4];

    if ((x4_glo >= cols / 4 || y_glo >= rows)
    return ;
    float4 value4=input4[y_glo * (cols / 4) + x4_glo];
    smem[threadIdx.y][threadIdx.x * 4 + 0]=value4.x;
    smem[threadIdx.y][threadIdx.x * 4 + 1]=value4.y;
    smem[threadIdx.y][threadIdx.x * 4 + 2]=value4.z;
    smem[threadIdx.y][threadIdx.x * 4 + 3]=value4.w;
    __syncthreads();

    float t_x=smem[threadIdx.x * 4 + 0][threadIdx.y];
    float t_y=smem[threadIdx.x * 4 + 1][threadIdx.y];
    float t_z=smem[threadIdx.x * 4 + 2][threadIdx.y];
    float t_w=smem[threadIdx.x * 4 + 3][threadIdx.y];
    float4 tran_value4=make_float4(t_x, t_y, t_z, t_z, t_w);
    __syncthreads();

    //交换行列
    int tran_x_glo = y_glo;
    int tran_y_glo_base = x4_glo * 4;
    int tran_cols=rows;
    int tran_rows=cols;
    if (tran_x_glo >= tran_cols || tran_y_glo_base >= tran_rows / 4)
    {}
    else {
    output[(tran_y_glo_base + 0) * rows + tran_x_glo]=t_x;
    output[(tran_y_glo_base + 1) * rows + tran_x_glo]=t_y;
    output[(tran_y_glo_base + 2) * rows + tran_x_glo]=t_z;
    output[(tran_y_glo_base + 3) * rows + tran_x_glo]=t_w;    
    }
}
