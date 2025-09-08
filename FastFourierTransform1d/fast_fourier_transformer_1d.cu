#include "../cudapacked.h"
//复数结构体
struct Complex {
    float real;  //实部
    float imag;  //虚部
};
//复数乘法(a+bi)*(c+di) = (ac-bd) + (ad+bc)i
__device__ Complex complex_mul(Complex a, Complex b) {
    Complex res;
    res.real=a.real * b.real - a.imag * b.imag;
    res.imag=a.real * b.imag + a.imag * b.real;
    return res;
}
//复数加法 (a+bi)+(c+di) = (a+c)+(b+d)i
__device__ Complex complex_add(Complex a, Complex b) {
    Complex res;
    res.real=a.real + b.real;
    res.imag=a.imag + b.imag;
    return res;
}
// 复数减法 (a+bi)-(c+di) = (a-c)+(b-d)i
__device__ Complex complex_sub(Complex a, Complex b) {
    Complex res;
    res.real=a.real - b.real;
    res.imag=a.imag - b.imag;
    return res;
}

//! 一维线程块 一维网格
//! 输入长度length必须是2的幂
template<int TIDSIZE>
__global__ void fast_fourier_transformer_1d(
    Complex *audio_1d, Complex *output, int length
) 
{
    int blo_x=threadIdx.x;
    int glo_x=blockIdx.x * TIDSIZE + blo_x;
    
    if (glo_x >= length) 
    return ;

    //初始化输出数组
    output[glo_x] = audio_1d[glo_x];
    //分阶段进行蝶形运算
    //总阶段数log2(length)
    for (int stage = 1; stage < length; stage *= 2) {
    //每个阶段组大小2*stage 每组2个蝶形单元
    int group_size = 2 * stage;
    
    //欧拉公式计算旋转因子 W = e^(-2πi * k / group_size) = cosθ - i*sinθ
    float angle = -2 * M_PI * (glo_x % stage) / group_size;
    Complex w;  // 旋转因子
    w.real=cosf(angle);
    w.imag=sinf(angle);

    //找到当前线程在组内的配对索引
    int pair_idx=glo_x + stage; //同组内的另一个元素索引
    //如果配对索引超出范围无需计算
    if (pair_idx >= length) 
    continue ;
    
    //读取当前元素和配对元素
    Complex x = output[glo_x]; //当前元素
    Complex y = output[pair_idx]; //配对元素

    //蝶形运算 y乘以旋转因子后与x进行加减
    Complex wy= complex_mul(w, y); //y * 旋转因子
    Complex new_x=complex_add(x, wy); //新的当前元素
    Complex new_y=complex_sub(x, wy); //新的配对元素

    //写回 确保所有线程读取完成
    __syncthreads();
    output[glo_x] = new_x;
    output[pair_idx] = new_y;
    __syncthreads();
    }
}
