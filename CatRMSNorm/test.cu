#include "cukernel.h"

void generate_random_half(half* h_data, int len) {
    std::mt19937 gen(40);
    // 均匀分布：[min_val, max_val]
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int i = 0; i < len; i++) {
        float rand_float = dist(gen);  // 生成高质量随机float
        h_data[i] = __float2half(rand_float);  // 转为half类型
    }
}

int main(int argc, char const *argv[])
{
    int dim = 7168;
    float eps = 1e-6f;

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCKSIZE = BLOCK_Y * BLOCK_X;
    dim3 block(BLOCK_X, BLOCK_Y);
    int GRID_X = (dim + BLOCKSIZE - 1) / BLOCKSIZE;
    int GRID_X_v3 = (dim + BLOCKSIZE * 4 - 1) / (BLOCKSIZE * 4);
    int GRID_Y = 1;
    dim3 grid(GRID_X, GRID_Y);
    dim3 grid_v3(GRID_X_v3, GRID_Y);

    half* h_x_in = new half[dim];
    half* h_res_in = new half[dim];
    half* h_weights = new half[dim];
    half* h_res_out = new half[dim];

    generate_random_half(h_x_in, dim);
    generate_random_half(h_res_in, dim);
    generate_random_half(h_weights, dim);

    half *d_x_in, *d_res_in, *d_weights, *d_res_out;
    cudaMalloc(&d_x_in, dim * sizeof(half));
    cudaMalloc(&d_res_in, dim * sizeof(half));
    cudaMalloc(&d_weights, dim * sizeof(half));
    cudaMalloc(&d_res_out, dim * sizeof(half));

    cudaMemcpy(d_x_in, h_x_in, dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_in, h_res_in, dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, dim * sizeof(half), cudaMemcpyHostToDevice);

    float* tmp_variance = nullptr;
    cudaMalloc(&tmp_variance, sizeof(float));
    cudaMemset(tmp_variance, 0.0f, sizeof(float));

    cudaEvent_t st, ed;
    cudaStream_t cu_str;
    cudaStreamCreateWithFlags(&cu_str, cudaStreamNonBlocking);
    cudaEventCreate(&st);
    cudaEventCreate(&ed);
    cudaEventRecord(st);
    CUDA_CHECK(cudaDeviceSynchronize());
    void* kernelArgs[] = {&d_x_in, &d_res_in, &dim, &eps, &d_weights, &d_res_out, &tmp_variance};
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v2<32, 16>,
        grid, block,
        kernelArgs
    );  
    /*
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v3<32, 16, 4>,
        grid_v3, block,
        kernelArgs
    ); 
    groups_cat_rmsnorm_v4<BLOCK_X, BLOCK_Y>(
        d_x_in, d_res_in, 
        dim, eps, d_weights,
        d_res_out, cu_str 
    );
    */
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(ed);
    cudaEventSynchronize(ed);
    float time;
    cudaEventElapsedTime(&time, st, ed);

    cudaMemcpy(h_res_out, d_res_out, dim * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_x_in, d_x_in, dim * sizeof(half), cudaMemcpyDeviceToHost);

    //std::cout<<"计时 "<<time<<" ms\n";
    int start = 5444;
    for (int i = start; i < start + 10; i++) {
        printf("x_out[%d] = %.4f \n", i, __half2float(h_x_in[i]));
        printf("res_out[%d] = %.4f\n", i, __half2float(h_res_out[i]));
    }

    delete[] h_x_in;
    delete[] h_res_in;
    delete[] h_weights;
    delete[] h_res_out;
    cudaFree(d_x_in);
    cudaFree(d_res_in);
    cudaFree(d_weights);
    cudaFree(d_res_out);

    return 0;
}
