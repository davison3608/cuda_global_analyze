#include "cukernel.h"

template<>
__global__ __launch_bounds__(1024) void stage_decode::flash_attn_v2_decode_v1
    <32, 32, 1>(
    const half* __restrict__ Q, //[1, 1, q_head_num, qk_head_dim]
    const half* __restrict__ K, //[1, max_seq_len, kv_head_num, qk_head_dim]
    const half* __restrict__ V, //[1, max_seq_len, kv_head_num, kv_lora_rank]
    half* Attn_out, //[q_head_num, splitk, kv_lora_rank + 1]
    int curr_seq_len, //当前有效缓存长度
    int q_head_num, //q总头数目
    int qk_head_dim,
    int kv_lora_rank,
    int splitk,
    float sm_scale,
    float logit_cap
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 32;
    constexpr int BLOCK_Z = 1;
    constexpr int BLOCK_SIZE = BLOCK_Z * BLOCK_Y * BLOCK_X;
    constexpr int WARP_NUMS = BLOCK_Z * BLOCK_Y;
    constexpr int WARP_SIZE = BLOCK_X;

    constexpr int BLOCK_H = BLOCK_Z;
    constexpr int BLOCK_N = BLOCK_Y * 1;

    constexpr int MASK = 0xffffffff;
    constexpr int Sf_len0 = WARP_SIZE / 2;
    constexpr int Sf_len1 = Sf_len0 / 2;
    constexpr int Sf_len2 = Sf_len1 / 2;
    constexpr int Sf_len3 = Sf_len2 / 2;
    constexpr int Sf_len4 = Sf_len3 / 2;

    struct layout {
    __device__ layout() noexcept = default;

    public:
    int blo_x {-1};
    int blo_y {-1}; //对应序列内token
    int blo_z {-1}; //对应组内头 始终为0
    
    int blk_x {-1}; //对应序列分片
    int blk_y {-1}; //对应头组
    
    __device__ void init() noexcept {
        this->blo_x = threadIdx.x;
        this->blo_y = threadIdx.y; 
        this->blo_z = 0; 

        this->blk_x = blockIdx.x; 
        this->blk_y = blockIdx.y; 
    }
    __device__ __forceinline__ int&warp_id() noexcept 
    { return this->blo_y; }
    __device__ __forceinline__ int&lane_id() noexcept
    { return this->blo_x; }
    };
    layout index;
    index.init();

    //Q K片段加载 V片段加载
    __shared__ half alignas(4) SM_q_load_dk[BLOCK_H][WARP_SIZE];
    __shared__ half alignas(4) SM_k_load_dk[BLOCK_N][WARP_SIZE];
    __shared__ half alignas(4) SM_v_load_dv[BLOCK_N][WARP_SIZE];

    //每个分组全局最值与指数和
    __shared__ float SM_group_emax[BLOCK_H];
    __shared__ float SM_group_esum[BLOCK_H];

    //加权区域(Dv_max_randk >= kv_lora_rank) 加权区域的缩放参数(在每个BLOCK_N迭代更新)
    constexpr int Dv_max_randk = 640;
    __shared__ half alignas(4) SM_acc[BLOCK_H][Dv_max_randk]; 
    __shared__ float SM_group_re_scale[BLOCK_H];

    struct SM_view {
    __device__ SM_view() noexcept = default;

    public:
    half (*sm_q)[WARP_SIZE] = nullptr;
    half (*sm_k)[WARP_SIZE] = nullptr;
    half (*sm_v)[WARP_SIZE] = nullptr;

    float* sm_emax {nullptr};
    float* sm_esum {nullptr};
    half (*sm_acc)[Dv_max_randk] = nullptr;

    //绑定片段内存地址
    __device__ __forceinline__ void SM_bind() noexcept {
        this->sm_q = SM_q_load_dk;
        this->sm_k = SM_k_load_dk;
        this->sm_v = SM_v_load_dv;

        this->sm_emax = SM_group_emax;
        this->sm_esum = SM_group_esum;
        this->sm_acc = SM_acc;
    }
    //首warp初始化全局维护与累加区域 
    __device__ void reset_e_acc(layout&index) noexcept {
    int&blo_y = index.blo_y;
    int&blo_x = index.blo_x;
    if (blo_y < BLOCK_H) {
        this->sm_emax[blo_y] = -INFINITY;
        this->sm_esum[blo_y] = __int2float_rn(0);

        #pragma unroll
        for (int i=0; i<Dv_max_randk/WARP_SIZE; i++)
        this->sm_acc[blo_y][i * WARP_SIZE + blo_x] = __int2half_rn(0);
    }
    this->block_sync();
    }
    __device__ __forceinline__ void block_sync() const 
    { __syncthreads(); } 
    __device__ __forceinline__ void warp_sync() const 
    { __syncwarp(MASK); } 
    };
    SM_view SM_load;
    SM_load.SM_bind();
    SM_load.reset_e_acc(index);

    struct BlockRange {
    __device__ BlockRange() = delete;
    __device__ BlockRange(int&blk_y, int&blk_x, int&curr_seq_len, int splitk) noexcept: 
        split_size((curr_seq_len + splitk - 1) / splitk), 
        split_start(blk_x * split_size),
        split_end(min(split_start + split_size, curr_seq_len)),
        group_head_st(blk_y * BLOCK_H),
        group_head_ed(group_head_st + BLOCK_H) {}
    
    public:
    //每个block处理的头范围 左闭右开
    const int group_head_st;
    const int group_head_ed;
    //每个block处理的splitk序列长度 左闭右开
    const int split_size;  
    const int split_start;
    const int split_end;
    };
    BlockRange range(index.blk_y, index.blk_x, curr_seq_len, splitk);

    struct Iter {
    __device__ Iter() noexcept = delete;    
    __device__ Iter(BlockRange&range, int&qk_head_dim, int&kv_lora_rank) noexcept:
        seq_iter((range.split_size + BLOCK_N - 1) / BLOCK_N),
        d_k_iter(qk_head_dim / WARP_SIZE),
        d_v_iter(kv_lora_rank / WARP_SIZE) {}

    public:
    //有效block计算序列迭代次数
    int seq_iter {-1};
    //Q K秩迭代次数 V秩迭代次数
    int d_k_iter {-1};
    int d_v_iter {-1};
    };
    Iter seq_d_iter(range, qk_head_dim, kv_lora_rank);

    //对于多余block提前退出
    if (range.split_start >= range.split_end)
        return;

    for (int step=0; step<seq_d_iter.seq_iter; step++) {
    //Q注意力头偏移 
    int offset_head = index.blk_y * BLOCK_H + index.blo_z;
    //当前split内的token偏移 全局token偏移
    int offset_token_l = step * BLOCK_N + index.warp_id();
    int offset_token_g = range.split_start + offset_token_l;

    //边界掩码
    bool mask_token = false;
    mask_token = (offset_token_g >= range.split_end)? true:false;

    //临时qk注意力矩阵首warp重置 
    __shared__ float alignas(4) SM_qk[BLOCK_H][BLOCK_N];
    if (index.warp_id() < BLOCK_H && index.lane_id() < BLOCK_N)
        SM_qk[index.warp_id()][index.lane_id()] = 0.0f;

    //每个warp持有单个元素共组成BLOCK_N局部向量
    float qk_i {0.0f};
    SM_load.block_sync();

    //qk内层迭代 所有warp进入
    #pragma unroll
    for (int n=0; n<seq_d_iter.d_k_iter; n++) {
    int offset_dk = n * WARP_SIZE + index.lane_id();

    //加载q注意力头片段
    int&sm_q_r = index.blo_z;
    int&sm_q_c = index.lane_id();
    if (index.warp_id() < BLOCK_H) 
    SM_load.sm_q[sm_q_r][sm_q_c] = Q[
        offset_head * qk_head_dim + 
        offset_dk
    ];

    //加载k序列片段
    int&sm_k_r = index.warp_id();
    int&sm_k_c = index.lane_id();
    SM_load.sm_k[sm_k_r][sm_k_c] = (mask_token)? __int2half_rn(0):K[
        offset_token_g * qk_head_dim + 
        offset_dk
    ];

    //等待qk片段
    SM_load.block_sync();

    //每个warp计算[1, 32] * [BLOCK_N//1, 32] 累加到qk缓存
    float qk_i_t = __half2float(SM_load.sm_q[sm_q_r][sm_q_c]) * 
        __half2float(SM_load.sm_k[sm_k_r][sm_k_c]);

    qk_i_t += __shfl_xor_sync(MASK, qk_i_t, Sf_len0);
    qk_i_t += __shfl_xor_sync(MASK, qk_i_t, Sf_len1);
    qk_i_t += __shfl_xor_sync(MASK, qk_i_t, Sf_len2);
    qk_i_t += __shfl_xor_sync(MASK, qk_i_t, Sf_len3);
    qk_i_t += __shfl_xor_sync(MASK, qk_i_t, Sf_len4);
    qk_i_t = __shfl_sync(MASK, qk_i_t, 0);
    
    if (index.lane_id() == 0)
        qk_i += qk_i_t;
    
    //本次内积不被下一次加载污染
    SM_load.block_sync();
    }

    //qk缩放 logitcap
    int&sm_qk_r = index.blo_z;
    int&sm_qk_c = index.warp_id();
    if (index.lane_id() == 0) {
        qk_i = __fmul_rn(qk_i, sm_scale);
        qk_i = __tanhf(__fdiv_rn(qk_i, logit_cap));
        qk_i = __fmul_rn(qk_i, logit_cap);

        qk_i = (mask_token)? -INFINITY:qk_i;
        SM_qk[sm_qk_r][sm_qk_c] = qk_i;
    }

    //等待qk原始矩阵
    SM_load.block_sync();

    //warp长度在qk低维迭代
    constexpr int WARP_ITER = (BLOCK_N + WARP_SIZE - 1) / WARP_SIZE;
    
    //online softmax 每个warp负责当前qk单头
    if (index.warp_id() < BLOCK_H) {
    float n_e_max {-INFINITY}; //多头最新局部最值
    float h_e_max {-INFINITY}; //多头局部最值
    float re_scale {0.0f}; //累加调整因子
    float h_e_sum {0.0f}; //局部指数和

    //每个头的局部最值由一个warp负责
    #pragma unroll
    for (int i=0; i<WARP_ITER; i++) {
    int offset_token = i * WARP_SIZE + index.lane_id();
    bool mask = (offset_token >= BLOCK_N)? true:false;
    
    float token = (mask)? -INFINITY:SM_qk[index.warp_id()][offset_token];
    float src = token;
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len0));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len1));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len2));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len3));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len4));
    src = __shfl_sync(MASK, src, 0);    

    if (index.lane_id() == 0)
        h_e_max = fmaxf(h_e_max, src);
    }
    //计算多头最新最值并覆盖到全局最值
    if (index.lane_id() == 0) {
        n_e_max = fmaxf(SM_load.sm_emax[index.warp_id()], h_e_max);
        re_scale = __expf(SM_load.sm_emax[index.warp_id()] - n_e_max);
        SM_load.sm_emax[index.warp_id()] = n_e_max;
    }
    
    //每个头的在线参数
    n_e_max = __shfl_sync(MASK, n_e_max, 0, WARP_SIZE);
    h_e_max = __shfl_sync(MASK, h_e_max, 0, WARP_SIZE);
    re_scale = __shfl_sync(MASK, re_scale, 0, WARP_SIZE);

    //qk注意力矩阵数值稳定
    #pragma unroll
    for (int i=0; i<WARP_ITER; i++) {
    int offset_token = i * WARP_SIZE + index.lane_id();
    bool mask = (offset_token >= BLOCK_N)? true:false;

    //注意这一步将原始qk的掩码token变为0.0f 后续不影响指数和
    float token = (mask)? -INFINITY:SM_qk[index.warp_id()][offset_token];
    token = __expf(token - n_e_max); 

    if (mask == false)
        SM_qk[index.warp_id()][offset_token] = token;
    }

    //qk矩阵的每个头保证稳定后统一读取
    SM_load.warp_sync();

    //每个头的局部指数和计算后 缩放旧指数和并累加
    #pragma unroll
    for (int i=0; i<WARP_ITER; i++) {
    int offset_token = i * WARP_SIZE + index.lane_id();
    bool mask = (offset_token >= BLOCK_N)? true:false;

    float token = (mask)? 0.0f:SM_qk[index.warp_id()][offset_token];
    float src = token;
    src += __shfl_xor_sync(MASK, src, Sf_len0);
    src += __shfl_xor_sync(MASK, src, Sf_len1);
    src += __shfl_xor_sync(MASK, src, Sf_len2);
    src += __shfl_xor_sync(MASK, src, Sf_len3);
    src += __shfl_xor_sync(MASK, src, Sf_len4);
    src = __shfl_sync(MASK, src, 0);    

    if (index.lane_id() == 0) 
        h_e_sum += src; 
    }
    //根据局部多头指数和累加到全局指数和
    if (index.lane_id() == 0) {
        float o_e_sum = SM_load.sm_esum[index.warp_id()];
        o_e_sum = __fadd_rn(o_e_sum * re_scale, h_e_sum);
        SM_load.sm_esum[index.warp_id()] = o_e_sum;
    }

    //重新覆盖本次序列片段的累加调整缩放因子
    if (index.lane_id() == 0)
        SM_group_re_scale[index.warp_id()] = re_scale;
    }
    
    //等待加权区域的调整因子
    SM_load.warp_sync();
    SM_load.block_sync();

    //v加权迭代 所有warp进入
    #pragma unroll
    for (int n=0; n<seq_d_iter.d_v_iter; n++) {
    int offset_dv = n * WARP_SIZE + index.lane_id();

    //加载V片段
    int&sm_v_r = index.warp_id();
    int&sm_v_c = index.lane_id();
    SM_load.sm_v[sm_v_r][sm_v_c] = (mask_token)? __int2half_rn(0):V[
        offset_token_g * kv_lora_rank + 
        offset_dv
    ];

    //等待v片段
    SM_load.block_sync();

    //qk与WARP_SIZE列的所有BLOCK_Y行乘积 首warp在公共边BLOCK_N迭代
    if (index.warp_id() < BLOCK_H) {
    float sum_pv_t {0.0f};

    #pragma unroll
    for (int bn=0; bn<BLOCK_N; bn++) {
        sum_pv_t += __fmul_rn(
            SM_qk[index.blo_z][bn],
            __half2float(SM_load.sm_v[bn][index.lane_id()])
        );
    }

    half o_acc = SM_load.sm_acc[index.blo_z][offset_dv];
    //根据本次缩放参数调整累加区域
    o_acc = __hmul_rn(__float2half_rn(SM_group_re_scale[index.blo_z]), o_acc);
    o_acc = __hadd_rn(o_acc, __float2half_rn(sum_pv_t));
    //写入到缓存本次迭代的对应列
    SM_load.sm_acc[index.blo_z][offset_dv] = o_acc;
    }

    //本次内积不被下一次加载污染
    SM_load.block_sync();
    }
    }

    //合并阶段附加参数 logsumexp
    float lse = __fadd_rn(
        SM_load.sm_emax[index.blo_z],
        __logf(SM_load.sm_esum[index.blo_z])
    );
    //写回到中间注意力输出 [q_head_num, splitk, kv_lora_rank]
    if (index.warp_id() < BLOCK_H) {
    //注意力头偏移 分片偏移
    int&offset_head = index.blk_y; //由于BLOCK_H固定为1 blk_y * BLOCK_H + blo_z
    int&offset_splitk = index.blk_x;
    
    #pragma unroll
    for (int n=0; n<seq_d_iter.d_v_iter; n++) {
    int offset_dv = n * WARP_SIZE + index.lane_id();

    Attn_out[
        offset_head * (splitk * (kv_lora_rank + 1)) +
        offset_splitk * (kv_lora_rank + 1) +
        offset_dv
    ] = SM_load.sm_acc[index.warp_id()][offset_dv];
    }

    Attn_out[
        offset_head * (splitk * (kv_lora_rank + 1)) +
        offset_splitk * (kv_lora_rank + 1) +
        kv_lora_rank
    ] = __float2half_rn(lse);
    }
}

extern "C" void stage_decode::v1_launch(
    half* Q, half* K, half* V, half* Attn_out,
    int curr_seq_len, int q_head_num, 
    int qk_head_dim, int kv_lora_rank,
    int splitk, float logit_cap
) noexcept
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 32;
    constexpr int BLOCK_Z = 1;
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    int grid_x = splitk;
    int grid_y = q_head_num / BLOCK_Z;
    dim3 grid(grid_x, grid_y);

    float sm_scale = 1.0f / sqrtf((float)qk_head_dim);
    void* args[] = {
        (void*)&Q, (void*)&K, (void*)&V, (void*)&Attn_out,
        (void*)&curr_seq_len, (void*)&q_head_num, 
        (void*)&qk_head_dim, (void*)&kv_lora_rank, 
        (void*)&splitk, (void*)&sm_scale, (void*)&logit_cap
    };
    using namespace stage_decode;
    CUDA_CHECK(cudaLaunchKernel(
        (void*)&flash_attn_v2_decode_v1<BLOCK_X, BLOCK_Y, BLOCK_Z>,
        grid, block, args, 
        0, cudaStreamDefault
    ));
    CUDA_CHECK(cudaGetLastError());
    return;
}
