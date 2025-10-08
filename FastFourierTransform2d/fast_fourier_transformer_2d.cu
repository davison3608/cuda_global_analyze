#include "../cudapacked.h"

// 复数结构体
struct Complex {
    float real;  // 实部
    float imag;  // 虚部
};

// 复数乘法 (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
__device__ Complex complex_mul(Complex a, Complex b) {
    Complex res;
    res.real = a.real * b.real - a.imag * b.imag;
    res.imag = a.real * b.imag + a.imag * b.real;
    return res;
}

// 复数加法 (a+bi)+(c+di) = (a+c)+(b+d)i
__device__ Complex complex_add(Complex a, Complex b) {
    Complex res;
    res.real = a.real + b.real;
    res.imag = a.imag + b.imag;
    return res;
}

// 复数减法 (a+bi)-(c+di) = (a-c)+(b-d)i
__device__ Complex complex_sub(Complex a, Complex b) {
    Complex res;
    res.real = a.real - b.real;
    res.imag = a.imag - b.imag;
    return res;
}

// 内部使用的一维傅里叶变换函数（行处理）
template<int TIDSIZE>
__device__ void fft_1d_row(Complex* data, int width) {
    int col = threadIdx.x + blockIdx.x * TIDSIZE;
    if (col >= width) return;

    // 对当前行的每个元素进行蝶形运算
    for (int stage = 1; stage < width; stage *= 2) {
    int group_size = 2 * stage;
    float angle = -2 * M_PI * (col % stage) / group_size;
    Complex w;
    w.real = cosf(angle);
    w.imag = sinf(angle);

    int pair_idx = col + stage;
    if (pair_idx >= width) 
        continue;

    // 当前线程处理的全局索引（行内列索引）
    int row = blockIdx.y;  // 每个块处理一行
    int idx = row * width + col;
    int pair_global_idx = row * width + pair_idx;

    Complex x = data[idx];
    Complex y = data[pair_global_idx];

    Complex wy = complex_mul(w, y);
    Complex new_x = complex_add(x, wy);
    Complex new_y = complex_sub(x, wy);
    __syncthreads();

    data[idx] = new_x;
    data[pair_global_idx] = new_y;
    __syncthreads();
    }
}

// 内部使用的一维傅里叶变换函数（列处理）
template<int TIDSIZE>
__device__ void fft_1d_col(Complex* data, int width, int height) {
    int row = threadIdx.x + blockIdx.x * TIDSIZE;
    if (row >= height) return;

    // 对当前列的每个元素进行蝶形运算
    for (int stage = 1; stage < height; stage *= 2) {
    int group_size = 2 * stage;
    float angle = -2 * M_PI * (row % stage) / group_size;
    Complex w;
    w.real = cosf(angle);
    w.imag = sinf(angle);

    int pair_idx = row + stage;
    if (pair_idx >= height) 
        continue;

    // 当前线程处理的全局索引（列内行索引）
    int col = blockIdx.y;  // 每个块处理一列
    int idx = row * width + col;
    int pair_global_idx = pair_idx * width + col;

    Complex x = data[idx];
    Complex y = data[pair_global_idx];

    Complex wy = complex_mul(w, y);
    Complex new_x = complex_add(x, wy);
    Complex new_y = complex_sub(x, wy);
    __syncthreads();

    data[idx] = new_x;
    data[pair_global_idx] = new_y;
    __syncthreads();
    }
}

//! 二维线程块 二维网络
//! 采用先行后列的变换策略 输入二维数组（按行优先存储）
//! 输入宽度width和高度height必须是2的幂
template<int TIDSIZE>
__global__ void fast_fourier_transformer_2d(
    Complex* input, Complex* output,   
    int width, int height         
) {
    // 首先将输入数据复制到输出数组
    int glo_x = threadIdx.x + blockIdx.x * TIDSIZE;
    int glo_y = blockIdx.y;
    int global_idx = glo_y * width + glo_x;
    
    if (glo_x < width && glo_y < height) {
        output[global_idx] = input[global_idx];
    }
    __syncthreads();

    // 对每一行执行一维傅里叶变换
    // 每个块处理一行的所有列
    fft_1d_row<TIDSIZE>(output, width);
    __syncthreads();

    // 对每一列执行一维傅里叶变换
    // 重新映射线程索引用于列处理
    glo_x = blockIdx.y;  // 列索引
    glo_y = threadIdx.x + blockIdx.x * TIDSIZE;  // 行索引
    if (glo_y < height && glo_x < width) {
        fft_1d_col<TIDSIZE>(output, width, height);
    }
}