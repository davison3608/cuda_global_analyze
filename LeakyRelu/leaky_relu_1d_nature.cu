#include "../cudapacked.h"
//! 一维线程块 一维网格
//! 一维网格线程尺寸覆盖向量长度
__global__ void leaky_relu_kernel(
    const float* input, float* output, int N
) 
{
    int blo_x=threadIdx.x;
    int blk_x=blockIdx.x;
    int glo_x=blk_x * blockDim.x + blo_x;
    
    float value=0.0f;
    if (glo_x >= N)
    return ;
    value=input[glo_x];
    value=(value > 0)? value:0.01 * value;
    output[glo_x]=value;
}

