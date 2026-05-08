#include "cukernel.h"

template<>
__global__ __launch_bounds__(256) void group_gemm_merge_v3_withpair
    <16, 16, 2>(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ pair_grid_x,
    int* __restrict__ pair_grid_y
)
{
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr int BLOCK_SIZE = 256; 
    constexpr int WARP_SIZE = 32;
    constexpr int Threads = 2;
    constexpr int Pairs = 2;

    int blo_x = threadIdx.x;
    int blo_y = threadIdx.y;
    int blo_idx = blo_y * BLOCK_X + blo_x;

    int warp_id = blo_idx / WARP_SIZE;
    int lane_id = blo_idx % WARP_SIZE;
    int blk_x = blockIdx.x; 
    int glo_idx = blk_x * BLOCK_SIZE + blo_idx; 

    int num_pairs = num_groups / Pairs;
    constexpr int Max_pairs = 4; //Max_pairs >= num_pairs
    __shared__ int sm_pair_x[Max_pairs];
    __shared__ int sm_pair_y[Max_pairs];

    bool mask_grid_size = true;
    mask_grid_size = (blo_idx >= num_pairs)? true:false;
    if (blo_idx < Max_pairs) {
        sm_pair_x[blo_idx] = (!mask_grid_size)? pair_grid_x[blo_idx]:-1;
        sm_pair_y[blo_idx] = (!mask_grid_size)? pair_grid_y[blo_idx]:-1;
    }
    __syncthreads(); 

    //确定所属的任务组偏移
    int pair_offset {-1};
    int pair_size {0};
    int2 pair_xy = make_int2(-1, -1);

    if (lane_id == 0) {
    int all_pair_size {0};

    #pragma unroll
    for (int i=0; i<num_pairs; i++) {
        //当前网格展平长度
        pair_xy.x = sm_pair_x[i];
        pair_xy.y = sm_pair_y[i];
        pair_size = pair_xy.y * pair_xy.x;
        //判断当前 global block index 是否落在该 group 区间内
        if (blk_x >= all_pair_size && blk_x < all_pair_size + pair_size) {
            pair_offset = i;
            break;
        }
        //目前为止的grid展平长度
        all_pair_size += pair_size;
    }
    }
    //每个warp内广播
    constexpr int MASK_ALL = 0xffffffff;
    pair_offset = __shfl_sync(MASK_ALL, pair_offset, 0);
    pair_size = __shfl_sync(MASK_ALL, pair_size, 0);
    pair_xy.x = __shfl_sync(MASK_ALL, pair_xy.x, 0);
    pair_xy.y = __shfl_sync(MASK_ALL, pair_xy.y, 0);

    //当前gird对应任务偏移基准
    int gemm_offset = pair_offset * Pairs;

    //当前grid对应的任务尺寸
    int M_cur, N_cur, K_cur {0};
    M_cur = M_[gemm_offset + 0];
    N_cur = N_[gemm_offset + 0];
    K_cur = K_[gemm_offset + 0];
    
    int M_next, N_next, K_next {0};
    M_next = M_[gemm_offset + 1];
    N_next = N_[gemm_offset + 1];
    K_next = K_[gemm_offset + 1];

    //当前grid对应任务指针
    const half* A_cur = (const half*)A_group[gemm_offset + 0];
    const half* B_cur = (const half*)B_group[gemm_offset + 0];
    half* C_cur = (half*)C_group[gemm_offset + 0];

    const half* A_next = (const half*)A_group[gemm_offset + 1];
    const half* B_next = (const half*)B_group[gemm_offset + 1];
    half* C_next = (half*)C_group[gemm_offset + 1];

    //跨步加载对应tile
    constexpr int K_len = 32;
    __shared__ half sm_A_cur[BLOCK_Y * Threads][K_len];
    __shared__ half sm_B_cur[K_len][BLOCK_X * Threads];
    
    __shared__ half sm_A_next[BLOCK_Y * Threads][K_len];
    __shared__ half sm_B_next[K_len][BLOCK_X * Threads];

    //原精度累加
    float sum_cur[Threads][Threads] {0.0f};
    float sum_next[Threads][Threads] {0.0f};

    //计算当前group之前的累计block数
    int prefix_blocks {0};
    if (lane_id == 0) {
    #pragma unroll
    for (int i = 0; i < pair_offset; ++i) {
        prefix_blocks += sm_pair_y[i] * sm_pair_x[i];
    }
    }
    //每个warp内广播
    prefix_blocks = __shfl_sync(MASK_ALL, prefix_blocks, 0);

    //当前grid中相对block展平偏移
    int blk_x_offset = blk_x - prefix_blocks;
    //分解为2D local block 坐标
    int local_blk_y = blk_x_offset / pair_xy.x;  
    int local_blk_x = blk_x_offset % pair_xy.x;
    //得到2D local thread 坐标
    int local_glo_y = local_blk_y * pair_xy.y + blo_y;
    int local_glo_x = local_blk_x * pair_xy.x + blo_x;

    //根据2D坐标找到C矩阵坐标基准
    int C_row_base = local_glo_y * Threads;
    int C_col_base = local_glo_x * Threads;

    //计算公共边跨步次数
    int iter_cur = (K_cur + K_len - 1) / K_len;
    int iter_next = (K_next + K_len - 1) / K_len;
    
    //选取最大迭代次数
    int iter_max = max(iter_cur, iter_next);
    #pragma unroll
    for (int it=0; it<iter_max; it++) {

    //加载当前group A 
    #pragma unroll
    for (int k=blo_x; k<K_len; k+=BLOCK_X) {
    //2x2区域的两行
    for (int dy=0; dy<Threads; dy++) {
        int sm_a_row = blo_y * Threads + dy;
        int sm_a_col = k;
        int a_row = C_row_base + dy;
        int a_col = it * K_len + k;
        
        bool mask_cur = a_row < M_cur && a_col < K_cur;
        bool maks_next = a_row < M_next && a_col < K_next;

        sm_A_cur[sm_a_row][sm_a_col] = (mask_cur)? 
            A_cur[a_row * K_cur + a_col]:__int2half_rn(0);
        sm_A_next[sm_a_row][sm_a_col] = (maks_next)?
            A_next[a_row * K_next + a_col]:__int2half_rn(0);
    }
    }

    //加载当前group B 0
    #pragma unroll
    for (int k=blo_x; k<K_len; k+=BLOCK_X) {
    //2x2区域的两列
    for (int dx=0; dx<Threads; dx++) {
        int sm_b_row = k;
        int sm_b_col = blo_x * Threads + dx;
        int b_row = it * K_len + k;
        int b_col = C_col_base + dx;

        bool mask_cur = b_row < K_cur && b_col < N_cur;
        bool maks_next = b_row < K_next && b_col < N_next;

        sm_B_cur[sm_b_row][sm_b_col] = (mask_cur)? 
            B_cur[b_row * N_cur + b_col]:__int2half_rn(0);
        sm_B_next[sm_b_row][sm_b_col] = (maks_next)?
            B_next[b_row * N_next + b_col]:__int2half_rn(0);
    }
    }

    //同步后片段内积
    __syncthreads(); 

    half _a_cur[Threads] {0.0f};
    half _b_cur[Threads] {0.0f};
    half _a_next[Threads] {0.0f};
    half _b_next[Threads] {0.0f};

    #pragma unroll
    for (int k=0; k<K_len; k++) {
    //线程对应2x2区域 
    #pragma unroll
    for (int dy=0; dy<Threads; dy++) {
        _a_cur[dy] = sm_A_cur[blo_y * Threads + dy][k];
        _a_next[dy] = sm_A_next[blo_y * Threads + dy][k];

        for (int dx=0; dx<Threads; dx++) {
            _b_cur[dx] = sm_B_cur[k][blo_x * Threads + dx];
            _b_next[dx] = sm_B_next[k][blo_x * Threads + dx];

            half val_0 = __hmul_rn(_a_cur[dy], _b_cur[dx]);
            sum_cur[dy][dx] += __half2float(val_0);

            half val_1 = __hmul_rn(_a_next[dy], _b_next[dx]);
            sum_next[dy][dx] += __half2float(val_1);
        }
    }
    }
    
    //下一次累加之前同步 
    __syncthreads(); 
    }

    //写回阶段
    #pragma unroll
    for (int dy=0; dy<Threads; dy++) {
    #pragma unroll
    for (int dx=0; dx<Threads; dx++) {
        //一次写进2x2区域
        int c_row = C_row_base + dy;
        int c_col = C_col_base + dx;
        if (c_row < M_cur && c_col < N_cur) {
            float source = __half2float(C_cur[c_row * N_cur + c_col]);
            float val = __fmul_rn(sum_cur[dy][dx], alpha) + __fmul_rn(source, beta);
            C_cur[c_row * N_cur + c_col] = __float2half_rn(val);
        }
        else if (c_row < M_next && c_col < N_next) {
            float source = __half2float(C_next[c_row * N_next + c_col]);
            float val = __fmul_rn(sum_next[dy][dx], alpha) + __fmul_rn(source, beta);
            C_next[c_row * N_next + c_col] = __float2half_rn(val);
        }
        else
            return;
    }
    }
}

extern "C" __host__ void group_gemm_merge_v3_withpair_launch(
    const void* A_group[], 
    const void* B_group[],
    void* C_group[], 
    const int* M_host, 
    const int* N_host, 
    const int* K_host, 
    const int num_groups,
    float&alpha,
    float&beta,
    cudaStream_t&cu_str
) noexcept
{
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr int Threads = 2;
    assert(num_groups % 2 == 0);
    constexpr dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(0, 1);

    int num_pairs = num_groups / 2;
    int pair_grid_x[num_pairs] = {0};
    int pair_grid_y[num_pairs] = {0};
    for (int p = 0; p < num_pairs; ++p) {
        int g0 = 2 * p;
        int g1 = 2 * p + 1;
        int gx0 = (N_host[g0] + BLOCK_X * Threads - 1) / (BLOCK_X * Threads);
        int gy0 = (M_host[g0] + BLOCK_Y * Threads - 1) / (BLOCK_Y * Threads);
        int gx1 = (N_host[g1] + BLOCK_X * Threads - 1) / (BLOCK_X * Threads);
        int gy1 = (M_host[g1] + BLOCK_Y * Threads - 1) / (BLOCK_Y * Threads);

        pair_grid_x[p] = std::max(gx0, gx1);
        pair_grid_y[p] = std::max(gy0, gy1);
        grid.x += pair_grid_x[p] * pair_grid_y[p];
    }

    int *pair_grid_x_dev,* pair_grid_y_dev = nullptr;
    int *M_dev,* N_dev,* K_dev = nullptr;
    const void **A_group_dev = nullptr;
    const void **B_group_dev = nullptr;
    void **C_group_dev = nullptr;

    cudaMallocAsync(&pair_grid_x_dev, num_pairs * sizeof(int), cu_str);
    cudaMallocAsync(&pair_grid_y_dev, num_pairs * sizeof(int), cu_str);
    cudaMallocAsync(&M_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&N_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&K_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&A_group_dev, num_groups * sizeof(void*), cu_str);
    cudaMallocAsync(&B_group_dev, num_groups * sizeof(void*), cu_str);
    cudaMallocAsync(&C_group_dev, num_groups * sizeof(void*), cu_str);

    cudaMemcpyAsync(pair_grid_x_dev, pair_grid_x, num_pairs * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(pair_grid_y_dev, pair_grid_y, num_pairs * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(M_dev, M_host, num_groups * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(N_dev, N_host, num_groups * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(K_dev, K_host, num_groups * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(A_group_dev, A_group, num_groups * sizeof(void*), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(B_group_dev, B_group, num_groups * sizeof(void*), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(C_group_dev, C_group, num_groups * sizeof(void*), cudaMemcpyHostToDevice, cu_str);

    void* args[] = {
        (void*)&A_group_dev, (void*)&B_group_dev, (void*)&C_group_dev,
        (void*)&M_dev, (void*)&N_dev, (void*)&K_dev, 
        (void*)&num_groups, 
        (void*)&alpha, (void*)&beta,
        (void*)&pair_grid_x_dev, (void*)&pair_grid_y_dev
    };
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStreamSynchronize(cu_str);
    CUDA_CHECK(cudaStreamBeginCapture(cu_str, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaLaunchKernel(
        (void*)group_gemm_merge_v3_withpair<BLOCK_X, BLOCK_Y, Threads>, 
        grid, block, 
        args, 0 , cu_str
    ));
    CUDA_CHECK(cudaStreamEndCapture(cu_str, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, cu_str));
    CUDA_CHECK(cudaStreamSynchronize(cu_str));

    cudaFreeAsync(pair_grid_x_dev, cu_str);
    cudaFreeAsync(pair_grid_y_dev, cu_str);
    cudaFreeAsync(M_dev, cu_str);
    cudaFreeAsync(N_dev, cu_str);
    cudaFreeAsync(K_dev, cu_str);
    cudaFreeAsync((void*)A_group_dev, cu_str);
    cudaFreeAsync((void*)B_group_dev, cu_str);
    cudaFreeAsync(C_group_dev, cu_str);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
}
