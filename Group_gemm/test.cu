#include "cukernel.h"

void generate_random_half(half* h_data, int len) {
    static std::random_device rd;               
    static std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int i = 0; i < len; i++) {
        float rand_float = dist(gen);
        h_data[i] = __float2half(rand_float);
    }
}

int main(int argc, char const *argv[])
{
    using arr_mnk = std::array<int, 3>;
    using vec_arr = std::vector<arr_mnk>;

    //配置参数
    const vec_arr configs {
        {1024, 256, 512}, //M N K
        {512, 128, 1024},
        {2048, 1024, 128},
        {256, 512, 256}
    };
    auto num_groups = configs.size();

    std::vector<half*> A_arr;
    std::vector<half*> B_arr;
    A_arr.resize(num_groups);
    B_arr.resize(num_groups);
    void* A_arr_dev[num_groups] {nullptr};
    void* B_arr_dev[num_groups] {nullptr};
    void* C_arr_dev[num_groups] {nullptr};

    cudaStream_t cu_str;
    cudaStreamCreateWithFlags(&cu_str, cudaStreamNonBlocking);
    //开辟内存
    for (int i=0; i<num_groups; i++) {
        auto&config_cur = configs[i];
        const int&M = config_cur[0];
        const int&N = config_cur[1];
        const int&K = config_cur[2];
        cudaMallocAsync(&A_arr_dev[i], M * K * sizeof(half), cu_str);
        cudaMallocAsync(&B_arr_dev[i], K * N * sizeof(half), cu_str);
        cudaMallocAsync(&C_arr_dev[i], M * N * sizeof(half), cu_str);
        A_arr[i] = new half[M * K];
        B_arr[i] = new half[N * K];
        generate_random_half(A_arr[i], M * K);
        generate_random_half(B_arr[i], N * K);
        cudaMemsetAsync(C_arr_dev[i], __int2half_rn(0), M * N * sizeof(half), cu_str);
    }
    cudaStreamSynchronize(cu_str);
    //填充到设备
    for (int i=0; i<num_groups; i++) {
        auto&config_cur = configs[i];
        const int&M = config_cur[0];
        const int&N = config_cur[1];
        const int&K = config_cur[2];
        cudaMemcpyAsync(
            A_arr_dev[i], A_arr[i], M * K * sizeof(half),
            cudaMemcpyHostToDevice, cu_str
        );
        cudaMemcpyAsync(
            B_arr_dev[i], B_arr[i], N * K * sizeof(half),
            cudaMemcpyHostToDevice, cu_str
        );
    }
    cudaStreamSynchronize(cu_str);

    //构造 M_, N_, K_ 数组
    std::vector<int> M_vec(num_groups);
    std::vector<int> N_vec(num_groups);
    std::vector<int> K_vec(num_groups);
    for (int i = 0; i < num_groups; ++i) {
        M_vec[i] = configs[i][0];
        N_vec[i] = configs[i][1];
        K_vec[i] = configs[i][2];
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    group_gemm_merge_v4_vecloadcache_launch(
        const_cast<const void**>(A_arr_dev),                     
        const_cast<const void**>(B_arr_dev),                     
        C_arr_dev,                     
        M_vec.data(),                  
        N_vec.data(),                  
        K_vec.data(),                  
        num_groups,                    
        alpha,                         
        beta,                          
        cu_str                         
    );

    cudaDeviceSynchronize();

    for (int i = 0; i < num_groups; ++i) {
        delete[] A_arr[i];
        delete[] B_arr[i];
        cudaFreeAsync(A_arr_dev[i], cu_str);
        cudaFreeAsync(B_arr_dev[i], cu_str);
        cudaFreeAsync(C_arr_dev[i], cu_str);
    }
    cudaStreamDestroy(cu_str);

    return 0;
}
