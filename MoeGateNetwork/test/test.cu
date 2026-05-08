#include "../cukernel.h"

static inline int N_tokens = 1;
static inline int n_routed_experts = 256;
static inline int topk_num = 8;
static inline int num_groups = 8;
static inline int topk_groups = 4;
static inline float route_scale = 1.0f;
static inline bool is_float16 = true;

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
    half* d_gating_x;
    float* d_score_bias;
    float* d_topk_weights;
    int* d_topk_ids;

    cudaMalloc(&d_gating_x, N_tokens * n_routed_experts * sizeof(half));
    cudaMalloc(&d_score_bias, n_routed_experts * sizeof(float));
    cudaMalloc(&d_topk_weights, N_tokens * topk_num * sizeof(float));
    cudaMalloc(&d_topk_ids, N_tokens * topk_num * sizeof(int));

    std::vector<half> h_gating(N_tokens * n_routed_experts);
    std::vector<float> h_bias(n_routed_experts, 0.0f);

    generate_random_half(h_gating.data(), h_gating.size());

    cudaMemcpy(d_gating_x, h_gating.data(),
               N_tokens * n_routed_experts * sizeof(half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_score_bias, h_bias.data(),
               n_routed_experts * sizeof(float),
               cudaMemcpyHostToDevice);

    GroupedTopkfused_v1_launch(
        (const void*)d_gating_x,
        d_score_bias,
        N_tokens,
        n_routed_experts,
        topk_num,
        num_groups,
        topk_groups,
        route_scale,
        d_topk_weights,
        d_topk_ids,
        is_float16
    );

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << "\n";
        exit(1);
    }

    cudaFree(d_gating_x);
    cudaFree(d_score_bias);
    cudaFree(d_topk_weights);
    cudaFree(d_topk_ids);

    cudaDeviceReset();
}

int main(int argc, char const *argv[])
{
    for (int i = 0; i < 10; i++) 
        test(i);
    printf("test down\n");
    return 0;
}


