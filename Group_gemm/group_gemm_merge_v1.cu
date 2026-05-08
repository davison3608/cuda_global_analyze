#include "cukernel.h"

template<>
__global__ __launch_bounds__(256) void group_gemm_merge_v1
    <16, 16>(
    const void* const A_group[], // [num_groups][M_i * K_i]
    const void* const B_group[], // [num_groups][K_i * N_i]
    void* C_group[], // [num_groups][M_i * N_i]
    const int* __restrict__ M_, // [num_groups]
    const int* __restrict__ N_, // [num_groups]
    const int* __restrict__ K_, // [num_groups]
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
)
{
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr int BLOCK_SIZE = 256; 
    constexpr int WARP_SIZE = 32;

    //auto block_grp = cooperative_groups::this_thread_block();
    //auto grid_grp = cooperative_groups::this_grid();
    int blo_x = threadIdx.x; //block_grp.thread_index().x;
    int blo_y = threadIdx.y; //block_grp.thread_index().y;
    int blo_idx = blo_y * BLOCK_X + blo_x; //block_grp.thread_rank();

    int warp_id = blo_idx / WARP_SIZE;
    int lane_id = blo_idx % WARP_SIZE;
    int blk_x = blockIdx.x; //grid_grp.block_index().x;
    int glo_idx = blk_x * BLOCK_SIZE + blo_idx; //grid_grp.thread_rank();

    constexpr int Max_Groups = 8; //Max_Groups >= num_groups
    __shared__ int sm_grid_x[Max_Groups];
    __shared__ int sm_grid_y[Max_Groups];

    bool mask_grid_size = true;
    mask_grid_size = (blo_idx >= num_groups)? true:false;
    if (blo_idx < Max_Groups) {
        sm_grid_x[blo_idx] = (!mask_grid_size)? grid_size_x[blo_idx]:-1;
        sm_grid_y[blo_idx] = (!mask_grid_size)? grid_size_y[blo_idx]:-1;
    }
    __syncthreads(); //block_grp.sync();

    //确定所属的任务偏移
    int group_offset {-1};
    int grid_size {0};
    int2 grid_xy = make_int2(-1, -1);

    if (lane_id == 0) {
    int all_grid_size {0};

    #pragma unroll
    for (int i=0; i<num_groups; i++) {
        //当前grid展平长度
        grid_xy.x = sm_grid_x[i];
        grid_xy.y = sm_grid_y[i];
        grid_size = grid_xy.y * grid_xy.x;
        //判断当前 global block index 是否落在该 group 区间内
        if (blk_x >= all_grid_size && blk_x < all_grid_size + grid_size) {
        group_offset = i;
        break;
        }
        //目前为止的grid展平长度
        all_grid_size += grid_size;
    }
    }
    //每个warp内广播
    constexpr uint MASK_ALL = 0xffffffff;
    group_offset = __shfl_sync(MASK_ALL, group_offset, 0);
    grid_size = __shfl_sync(MASK_ALL, grid_size, 0);
    grid_xy.x = __shfl_sync(MASK_ALL, grid_xy.x, 0);
    grid_xy.y = __shfl_sync(MASK_ALL, grid_xy.y, 0);

    //当前grid对应的任务尺寸
    int M_cur, N_cur, K_cur {0};
    M_cur = M_[group_offset];
    N_cur = N_[group_offset];
    K_cur = K_[group_offset];

    //当前grid对应任务指针
    const half* A_cur = (const half*)A_group[group_offset];
    const half* B_cur = (const half*)B_group[group_offset];
    half* C_cur = (half*)C_group[group_offset];

    //跨步加载对应tile
    constexpr int K_len = 32;
    __shared__ half sm_A_i[BLOCK_Y][K_len];
    __shared__ half sm_B_i[K_len][BLOCK_X];

    //原精度累加
    float sum_i {0.0f};

    //计算当前group之前的累计block数
    int prefix_blocks {0};
    if (lane_id == 0) {
    #pragma unroll
    for (int i = 0; i < group_offset; ++i) {
        prefix_blocks += sm_grid_y[i] * sm_grid_x[i];
    }
    }
    //每个warp内广播
    prefix_blocks = __shfl_sync(MASK_ALL, prefix_blocks, 0);

    //当前grid中相对block展平偏移
    int blk_x_offset = blk_x - prefix_blocks;
    //分解为2D local block 坐标
    int local_blk_y = blk_x_offset / grid_xy.x;  
    int local_blk_x = blk_x_offset % grid_xy.x;

    //计算公共边跨步次数
    int iter_i = (K_cur + K_len - 1) / K_len;
    #pragma unroll
    for (int it=0; it<iter_i; it++) {

    //加载当前group A
    #pragma unroll
    for (int k=blo_x; k<K_len; k+=BLOCK_X) {
        int&sm_a_row = blo_y;
        int&sm_a_col = k;
        int a_row = local_blk_y * BLOCK_Y + blo_y;
        int a_col = it * K_len + k;
        if (a_row < M_cur && a_col < K_cur) 
            sm_A_i[sm_a_row][sm_a_col] = A_cur[a_row * K_cur + a_col];
        else 
            sm_A_i[sm_a_row][sm_a_col] = __int2half_rn(0);
    }

    //加载当前group B
    #pragma unroll
    for (int k=blo_y; k<K_len; k+=BLOCK_Y) {
        int&sm_b_row = k;
        int&sm_b_col = blo_x;
        int b_row = it * K_len + k;
        int b_col = local_blk_x * BLOCK_X + blo_x;
        if (b_row < K_cur && b_col < N_cur)
            sm_B_i[sm_b_row][sm_b_col] = B_cur[b_row * N_cur + b_col];
        else
            sm_B_i[sm_b_row][sm_b_col] = __int2half_rn(0);
    }

    //同步后片段内积
    __syncthreads(); //block_grp.sync();

    float t_sum_i {0.0f};
    #pragma unroll
    for (int k=0; k<K_len; k++) {
        half t_val = __hmul_rn(sm_A_i[blo_y][k], sm_B_i[k][blo_x]);
        t_sum_i += __half2float(t_val);
    }

    //下一次累加之前同步 
    sum_i = __fadd_rn(sum_i, t_sum_i);
    __syncthreads(); //block_grp.sync();
    }
    
    //写回阶段
    if (local_blk_y < M_cur && local_blk_x < N_cur) {
        float source = __half2float(C_cur[local_blk_y * N_cur + local_blk_x]);
        float val = __fmul_rn(sum_i, alpha) + __fmul_rn(source, beta);
        C_cur[local_blk_y * N_cur + local_blk_x] = __float2half_rn(val);
    }
    else
        return;
}

extern "C" __host__ void group_gemm_merge_v1_launch(
    const void* A_group[], // [num_groups][M_i * K_i]
    const void* B_group[], // [num_groups][K_i * N_i]
    void* C_group[], // [num_groups][M_i * N_i]
    const int* M_host, // [num_groups]
    const int* N_host, // [num_groups]
    const int* K_host, // [num_groups]
    const int num_groups,
    float&alpha,
    float&beta,
    cudaStream_t&cu_str
) noexcept
{
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(0, 1);

    int grid_size_x[num_groups] = {0};
    int grid_size_y[num_groups] = {0};
    for (int i = 0; i < num_groups; ++i) {
        int col_block = (N_host[i] + BLOCK_X - 1) / BLOCK_X;
        int row_block = (M_host[i] + BLOCK_Y - 1) / BLOCK_Y;
        grid_size_x[i] = col_block;
        grid_size_y[i] = row_block;
        grid.x += row_block * col_block;
    }

    int* grid_size_x_dev,* grid_size_y_dev = nullptr;
    int* M_dev,* N_dev,* K_dev = nullptr;
    const void **A_group_dev = nullptr;
    const void **B_group_dev = nullptr;
    void **C_group_dev = nullptr;

    cudaMallocAsync(&grid_size_x_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&grid_size_y_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&M_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&N_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&K_dev, num_groups * sizeof(int), cu_str);
    cudaMallocAsync(&A_group_dev, num_groups * sizeof(void*), cu_str);
    cudaMallocAsync(&B_group_dev, num_groups * sizeof(void*), cu_str);
    cudaMallocAsync(&C_group_dev, num_groups * sizeof(void*), cu_str);

    cudaMemcpyAsync(grid_size_x_dev, grid_size_x, num_groups * sizeof(int), cudaMemcpyHostToDevice, cu_str);
    cudaMemcpyAsync(grid_size_y_dev, grid_size_y, num_groups * sizeof(int), cudaMemcpyHostToDevice, cu_str);
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
        (void*)&grid_size_x_dev, (void*)&grid_size_y_dev
    };
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStreamSynchronize(cu_str);
    CUDA_CHECK(cudaStreamBeginCapture(cu_str, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaLaunchKernel(
        (void*)group_gemm_merge_v1<BLOCK_X, BLOCK_Y>, 
        grid, block, 
        args, 0 , cu_str
    ));
    CUDA_CHECK(cudaStreamEndCapture(cu_str, &graph));
    CUDA_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, cu_str));
    CUDA_CHECK(cudaStreamSynchronize(cu_str));

    cudaFreeAsync(grid_size_x_dev, cu_str);
    cudaFreeAsync(grid_size_y_dev, cu_str);
    cudaFreeAsync(M_dev, cu_str);
    cudaFreeAsync(N_dev, cu_str);
    cudaFreeAsync(K_dev, cu_str);
    cudaFreeAsync((void*)A_group_dev, cu_str);
    cudaFreeAsync((void*)B_group_dev, cu_str);
    cudaFreeAsync(C_group_dev, cu_str);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
}
