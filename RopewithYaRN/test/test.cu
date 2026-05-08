#include "cukernel.h"
//#include <pybind11/pybind11.h>
//#include <pybind11/embed.h>
//#include <pybind11/stl.h> 

static inline int max_position = 128000;
static inline int seq_len = 1;
static inline int num_heads = 64;
static inline int rotary_dim = 64;

static inline int num_head_mla = 32
static inline int nope_dim_mla = 128  
static inline int rope_dim_mla = 64  
static inline int kv_lora_rank = 512 

static inline num_head_indexer = 64
static inline head_dim_indexer = 128
static inline nope_dim_indexer = head_dim_indexer - rotary_dim
static inline rope_dim_indexer = rotary_dim  

static inline int totall_test = 1;

static void generate_random_half(half* h_data, int len) {
    static std::random_device rd;               
    static std::mt19937 gen(rd());              
    static std::uniform_real_distribution<float> dist(0.0f, 10.0f);

    for (int i = 0; i < len; ++i) {
        float rand_val = dist(gen);
        h_data[i] = __float2half(rand_val);
    }
}

void test(int start_pos) {
    std::vector<half> cos_sin_cache(max_position * rotary_dim);
    generate_random_half(cos_sin_cache.data(), max_position * rotary_dim);

    //gptj风格qk
    std::vector<std::vector<half>> qk_pe_gpt(2);
    qk_pe_gpt[0].resize(seq_len * num_heads * rotary_dim);
    qk_pe_gpt[1].resize(seq_len * rotary_dim);
    generate_random_half(qk_pe_gpt[0].data(), seq_len * num_heads * rotary_dim);
    generate_random_half(qk_pe_gpt[1].data(), seq_len * rotary_dim);
    //neox风格qk
    std::vector<std::vector<half>> qk_pe_nex(2);
    qk_pe_nex[0].resize(seq_len * num_heads * rotary_dim);
    qk_pe_nex[1].resize(seq_len * rotary_dim);
    generate_random_half(qk_pe_nex[0].data(), seq_len * num_heads * rotary_dim);
    generate_random_half(qk_pe_nex[1].data(), seq_len * rotary_dim);
    
    half* cos_sin_cache_dev = nullptr;
    half* qk_pe_gpt_dev[2] {nullptr};
    half* qk_pe_nex_dev[2] {nullptr};

    ssize_t cache_size = max_position * rotary_dim * sizeof(half);
    ssize_t q_size = seq_len * num_heads * rotary_dim * sizeof(half);
    ssize_t k_size = seq_len * 1 * rotary_dim * sizeof(half);

    cudaMalloc(&cos_sin_cache_dev, cache_size);
    cudaMalloc(&qk_pe_gpt_dev[0], q_size);
    cudaMalloc(&qk_pe_gpt_dev[1], k_size);
    cudaMalloc(&qk_pe_nex_dev[0], q_size);
    cudaMalloc(&qk_pe_nex_dev[1], k_size);
    cudaMemcpy(cos_sin_cache_dev, cos_sin_cache.data(), cache_size, cudaMemcpyHostToDevice);
    cudaMemcpy(qk_pe_gpt_dev[0], qk_pe_gpt[0].data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(qk_pe_gpt_dev[1], qk_pe_gpt[1].data(), k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(qk_pe_nex_dev[0], qk_pe_nex[0].data(), q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(qk_pe_nex_dev[1], qk_pe_nex[1].data(), k_size, cudaMemcpyHostToDevice);

    int* positions = new int[seq_len];
    for (int i=0; i<seq_len; i++)
        positions[i] = start_pos + i;
    int* d_positions = nullptr;
    cudaMalloc(&d_positions, seq_len * sizeof(int));
    cudaMemcpy(d_positions, positions, seq_len * sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());
    
    gptj_style::rope_withyarn_gptj_v1_launch(
        cos_sin_cache_dev, d_positions,
        qk_pe_gpt_dev[0], qk_pe_gpt_dev[1],  
        seq_len, num_heads, rotary_dim, stream
    );
    
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    neox_style::rope_withyarn_neox_v1_launch(
        cos_sin_cache_dev, d_positions,
        qk_pe_nex_dev[0], qk_pe_nex_dev[1],  
        seq_len, num_heads, rotary_dim, stream
    );

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaFree(cos_sin_cache_dev);
    cudaFree(qk_pe_gpt_dev[0]);
    cudaFree(qk_pe_gpt_dev[1]);
    cudaFree(qk_pe_nex_dev[0]);
    cudaFree(qk_pe_nex_dev[1]);
    cudaFree(d_positions);
    delete[] positions;
}

int main(int argc, char const *argv[])
{
    for (int t=0; t<totall_test*seq_len; t+=seq_len) {
        test(t);
        //printf("迭代次数 %d start_pos %d \n", t/seq_len, t);
    }
    return 0;
}
