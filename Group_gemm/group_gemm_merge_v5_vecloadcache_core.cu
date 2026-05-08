#include "cukernel.h"

template<>
__global__ __launch_bounds__(256) void group_gemm_merge_v5_vecloadcache_core
    <16, 16, 2, 2>(
    const void* const A_group[], 
    const void* const B_group[], 
    void* C_group[], 
    const int* __restrict__ M_, 
    const int* __restrict__ N_, 
    const int* __restrict__ K_, 
    const int num_groups,
    const float alpha,
    const float beta,
    int* __restrict__ grid_size_x,
    int* __restrict__ grid_size_y
)
{
    using namespace nvcuda;
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr int BLOCK_SIZE = 256; 
    constexpr int WARP_SIZE = 32;
    constexpr int Threads = 2;
    constexpr int Vec_len = 2;

    int blo_x = threadIdx.x; 
    int blo_y = threadIdx.y; 
    int blo_idx = blo_y * BLOCK_X + blo_x; 

    int warp_id = blo_idx / WARP_SIZE;
    int lane_id = blo_idx % WARP_SIZE;
    int blk_x = blockIdx.x; 
    int glo_idx = blk_x * BLOCK_SIZE + blo_idx;

    constexpr int Max_Groups = 8; //Max_Groups >= num_groups
    __shared__ int sm_grid_x[Max_Groups];
    __shared__ int sm_grid_y[Max_Groups];

    bool mask_grid_size = true;
    mask_grid_size = (blo_idx >= num_groups)? true:false;
    if (blo_idx < Max_Groups) {
        sm_grid_x[blo_idx] = (!mask_grid_size)? grid_size_x[blo_idx]:-1;
        sm_grid_y[blo_idx] = (!mask_grid_size)? grid_size_y[blo_idx]:-1;
    }
    __syncthreads(); 

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

    //向量化加载
    int K_cur_vec, N_cur_vec {0};
    K_cur_vec = K_cur / Vec_len;
    N_cur_vec = N_cur / Vec_len;

    //当前grid对应任务指针
    const half* A_cur = (const half*)A_group[group_offset];
    const half* B_cur = (const half*)B_group[group_offset];
    half* C_cur = (half*)C_group[group_offset];

    const half2* A_cur2 = reinterpret_cast<const half2*>(A_cur);
    half2* C_cur2 = reinterpret_cast<half2*>(C_cur);

    //跨步加载对应tile
    constexpr int Cache = 2;
    constexpr int K_len = 64;
    constexpr int K_vec = K_len / Vec_len;
    __shared__ alignas(16) half sm_A_i[Cache][BLOCK_Y * Threads][K_len];
    __shared__ alignas(16) half sm_B_i[Cache][K_len][BLOCK_X * Threads];

    //原精度累加
    __shared__ float sm_C_frag[BLOCK_Y * Threads][BLOCK_X * Threads];

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
    //得到2D local thread 坐标
    int local_glo_y = local_blk_y * grid_xy.y + blo_y;
    int local_glo_x = local_blk_x * grid_xy.x + blo_x;

    //根据2D坐标找到C矩阵坐标基准
    int C_row_base = local_glo_y * Threads;
    int C_col_base = local_glo_x * Threads;

    //缓冲区切换索引 
    int curr, next {-1};
    curr = 0;
    next = 1;

    //预加载curr 保证第一次计算有数据
    int first_n=0;
    
    //预加载当前group A
    #pragma unroll
    for (int k=blo_x; k<K_vec; k+=BLOCK_X) {
    //2x2区域的两行
    for (int dy=0; dy<Threads; dy++) {
        int sm_a_row = blo_y * Threads + dy;
        int sm_a_col = k * Vec_len;
        int a_row = C_row_base + dy;
        int a_col = first_n * K_len + k;

        bool mask_cur = a_row < M_cur && a_col < K_cur_vec;
        half2 vec_a = (mask_cur)? A_cur2[a_row * K_cur_vec + a_col]:
            make_half2(__int2half_rn(0), __int2half_rn(0));
        sm_A_i[curr][sm_a_row][sm_a_col + 0] = vec_a.x;
        sm_A_i[curr][sm_a_row][sm_a_col + 1] = vec_a.y;
    }
    }

    //预加载当前group B
    #pragma unroll
    for (int k=blo_x; k<K_vec; k+=BLOCK_X) {
    //2x2区域的两列
    for (int dx=0; dx<Threads; dx++) {
        int sm_b_row = k * Vec_len;
        int sm_b_col = blo_x * Threads + dx;
        int b_row_base = first_n * K_len;
        int b_row = b_row_base + k;
        int b_col = C_col_base + dx;

        bool mask_cur = b_row < K_cur_vec && b_col < N_cur;
        half vec_b_x = (mask_cur)? B_cur[(b_row_base + k * Vec_len + 0) * N_cur + b_col]:
            __int2half_rn(0);
        half vec_b_y = (mask_cur)? B_cur[(b_row_base + k * Vec_len + 1) * N_cur + b_col]:
            __int2half_rn(0);
        sm_B_i[curr][sm_b_row + 0][sm_b_col] = vec_b_x;
        sm_B_i[curr][sm_b_row + 1][sm_b_col] = vec_b_y;
    }
    }

    //预加载完成
    __syncthreads();

    //tensor core片段
    constexpr int core_y_warps = (BLOCK_Y * Threads) / 16;
    constexpr int core_x_warps = (BLOCK_X * Threads) / 16;
    constexpr int core_warps = core_y_warps * core_x_warps;
    constexpr int core_iter = K_len / 16;
    
    //从参与warp中分解局部2D坐标
    int warp_y = lane_id / core_warps;
    int warp_x = lane_id % core_warps;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f); 
    
    //计算公共边跨步次数 
    int iter_i = (K_cur + K_len - 1) / K_len;
    #pragma unroll
    for (int it=0; it<iter_i; it++) {

    //从预加载开始进行内积
    if (lane_id < core_warps) {
    #pragma unroll
    for (int ci=0; ci<core_iter; ci++) {
    //从share中填充片段
    wmma::load_matrix_sync(a_frag, &sm_A_i[curr][warp_y * 16][ci * 16], K_len);
    wmma::load_matrix_sync(b_frag, &sm_B_i[curr][ci * 16][warp_x * 16], K_len);
    //迭代原地累加
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    }
    //无需同步 异步加载下一次片段

    //除去预加载
    if (it < iter_i - 1) {
    //加载当前group A
    #pragma unroll
    for (int k=blo_x; k<K_vec; k+=BLOCK_X) {
    //2x2区域的两行
    for (int dy=0; dy<Threads; dy++) {
        int sm_a_row = blo_y * Threads + dy;
        int sm_a_col = k * Vec_len;
        int a_row = C_row_base + dy;
        int a_col = it * K_len + k;

        bool mask_cur = a_row < M_cur && a_col < K_cur_vec;
        half2 vec_a = (mask_cur)? A_cur2[a_row * K_cur_vec + a_col]:
            make_half2(__int2half_rn(0), __int2half_rn(0));
        sm_A_i[next][sm_a_row][sm_a_col + 0] = vec_a.x;
        sm_A_i[next][sm_a_row][sm_a_col + 1] = vec_a.y;
    }
    }

    //加载当前group B
    #pragma unroll
    for (int k=blo_x; k<K_vec; k+=BLOCK_X) {
    //2x2区域的两列
    for (int dx=0; dx<Threads; dx++) {
        int sm_b_row = k * Vec_len;
        int sm_b_col = blo_x * Threads + dx;
        int b_row_base = it * K_len;
        int b_row = b_row_base + k;
        int b_col = C_col_base + dx;

        bool mask_cur = b_row < K_cur_vec && b_col < N_cur;
        half vec_b_x = (mask_cur)? B_cur[(b_row_base + k * Vec_len + 0) * N_cur + b_col]:
            __int2half_rn(0);
        half vec_b_y = (mask_cur)? B_cur[(b_row_base + k * Vec_len + 1) * N_cur + b_col]:
            __int2half_rn(0);
        sm_B_i[next][sm_b_row + 0][sm_b_col] = vec_b_x;
        sm_B_i[next][sm_b_row + 1][sm_b_col] = vec_b_y;
    }
    }

    //为下一次片段内积同步
    __syncthreads(); 
    }

    //切换缓冲区
    int temp = curr;
    curr = next;
    next = temp;
    }

    //片段返回到share
    if (lane_id < core_warps) {
    wmma::store_matrix_sync(&sm_C_frag[warp_y * 16][warp_x * 16], c_frag, BLOCK_X * Threads, wmma::mem_row_major);
    }
    __syncthreads();

    //线程级别写回
    #pragma unroll
    for (int dy=0; dy<Threads; dy++) {
        //一次写进2x2区域
        int c_row = C_row_base + dy;
        int c_col = C_col_base;
        
        bool mask_cur = c_row < M_cur && c_col < N_cur_vec;
        float2 source = (mask_cur)? __half22float2(C_cur2[c_row * N_cur_vec + c_col]):
            make_float2(0.0f, 0.0f);
        float2 val = make_float2(
            __fmul_rn(sm_C_frag[blo_y * Threads + dy][blo_x * Threads + 0], alpha)
             + __fmul_rn(source.x, beta),
            __fmul_rn(sm_C_frag[blo_y * Threads + dy][blo_x * Threads + 1], alpha)
             + __fmul_rn(source.y, beta)
        );
        if (mask_cur)
            C_cur2[c_row * N_cur_vec + c_col] = __float22half2_rn(val);
    }
}

extern "C" __host__ void group_gemm_merge_v5_vecloadcache_core_launch(
    const void* A_group[], const void* B_group[], void* C_group[], 
    const int* M_host, const int* N_host, const int* K_host, 
    const int num_groups, float&alpha, float&beta,
    cudaStream_t&cu_str
) noexcept
{
    constexpr int BLOCK_X = 16; 
    constexpr int BLOCK_Y = 16; 
    constexpr int Threads = 2;
    constexpr int Vec_len = 2;
    constexpr dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(0, 1);

    int grid_size_x[num_groups] = {0};
    int grid_size_y[num_groups] = {0};
    for (int i = 0; i < num_groups; ++i) {
        int col_block = (N_host[i] + BLOCK_X * Threads - 1) / (BLOCK_X * Threads);
        int row_block = (M_host[i] + BLOCK_Y * Threads - 1) / (BLOCK_Y * Threads);
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
        (void*)group_gemm_merge_v5_vecloadcache_core<BLOCK_X, BLOCK_Y, Threads, Vec_len>, 
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
