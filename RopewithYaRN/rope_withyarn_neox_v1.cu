#include "cukernel.h"

template<>
__global__ __launch_bounds__(512) void neox_style::rope_withyarn_neox_v1
    <32, 16, 512>(
    half* __restrict__ cos_sin_cache, //[max_position_embeddings, rotary_dim]
    int* __restrict__ positions,      //[batch_size, seq_len=1]
    half* __restrict__ query,         //[batch_size, seq_len=1, num_heads=32, dim]
    half* __restrict__ key,           //[batch_size, seq_len=1, num_heads=1, dim]
    int q_seq_stride, int q_head_stride,  
    int k_seq_stride, int k_head_stride, 
    int seq_len, int num_heads,                        
    int rotary_dim
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCK_SIZE = 512;
    constexpr int WARP_NUMS = BLOCK_Y;
    constexpr int WARP_SIZE = BLOCK_X;
    constexpr int WARP_Shfl = WARP_SIZE / 2;
    constexpr int MASK = 0xFFFFFFFF;

    struct layout {
    __device__ layout() noexcept = default;

    public:
    int32_t blo_x {-1}; //对应dim
    int32_t blo_y {-1}; //对应头数目
    int32_t blk_x {-1}; //对应序列长度
    int32_t blk_y {-1}; //对应头组
        
    __device__ void init() noexcept {
        this->blo_x = threadIdx.x;
        this->blo_y = threadIdx.y;
        this->blk_x = blockIdx.x; 
        this->blk_y = 0; 
    }
    __device__ __forceinline__ int&warp_id() 
    { return this->blo_y; }
    __device__ __forceinline__ int&lane_id() 
    { return this->blo_x; }
    };
    layout index; 
    index.init();

    //确定序列长度的绝对位置 
    struct cos_sin_pos {
    __device__ cos_sin_pos() noexcept = default;
    
    public:
    int seq_len_offset {-1};    //token相对位置
    int seq_len_pos {-1};       //token绝对位置
    int cache_len {-1};         //cossin缓存低维长度
    half* cos_cache {nullptr};  //cos缓存起始
    half* sin_cache {nullptr};  //sin缓存起始
    };
    cos_sin_pos pos_cs;

    if (index.blk_x >= seq_len)
        return;
    pos_cs.seq_len_offset = index.blk_x;
    pos_cs.seq_len_pos = positions[pos_cs.seq_len_offset];

    //确定cos sin缓存地址
    half *cache_tmp = cos_sin_cache + pos_cs.seq_len_pos * rotary_dim;
    pos_cs.cache_len = rotary_dim / 2;
    pos_cs.cos_cache = cache_tmp;
    pos_cs.sin_cache = cache_tmp + pos_cs.cache_len;

    __shared__ alignas(4) half SM_cos[WARP_SIZE];
    __shared__ alignas(4) half SM_sin[WARP_SIZE];

    __shared__ alignas(4) half SM_q[WARP_NUMS][WARP_SIZE];
    __shared__ alignas(4) half SM_q_rotate[WARP_NUMS][WARP_SIZE];

    __shared__ alignas(4) half SM_k[WARP_SIZE];
    __shared__ alignas(4) half SM_k_rotate[WARP_SIZE];

    struct SM_view {
    __device__ SM_view() noexcept = default;

    public:
    half* sm_cos = nullptr;
    half* sm_sin = nullptr;
    half (*sm_q)[WARP_SIZE] = nullptr;
    half (*sm_q_rot)[WARP_SIZE] = nullptr;
    half* sm_k = nullptr;
    half* sm_k_rot = nullptr;

    __device__ void SM_bind() noexcept {
        this->sm_cos = SM_cos;
        this->sm_sin = SM_sin;
        this->sm_q = SM_q;
        this->sm_q_rot = SM_q_rotate;
        this->sm_k = SM_k;
        this->sm_k_rot = SM_k_rotate;
    }
    __device__ __forceinline__ void block_sync() const
    { __syncthreads(); }
    __device__ __forceinline__ void warp_sync() const
    { __syncwarp(MASK); }
    };
    SM_view SM_load;
    SM_load.SM_bind();

    //block迭代次数
    struct iters {
    __device__ iters() noexcept {}
    
    int head_iter {-1};  //query头迭代
    int dim_iter {-1};   //query低维迭代
    };
    iters q_hd_iter;

    q_hd_iter.head_iter = num_heads / WARP_NUMS;
    q_hd_iter.dim_iter = rotary_dim / WARP_SIZE;

    //layout行掩码
    struct Mask_layout {
    __device__ Mask_layout() noexcept = default;
    __device__ Mask_layout(layout&_layout) noexcept {
        this->mask_w = (_layout.warp_id() == 0)? true:false;
        this->mask_h = (_layout.blk_y == 0)? true:false;
    }

    public:    
    bool mask_w {false}; //首行warp
    bool mask_h {false}; //首行block
    };
    Mask_layout mask_wh(index);

    #pragma unroll
    for (int h=0; h<q_hd_iter.head_iter; h++) {
    //首次加载key
    bool first_h = (h == 0)? true:false;
    
    #pragma unroll
    for (int d=0; d<q_hd_iter.dim_iter; d++) {    
    //三角缓存仅rotary_dim半长度
    bool mask_sf_load = (index.lane_id() < WARP_Shfl)? true:false;
    int&sm_r = index.warp_id();
    int&sm_c = index.lane_id();

    if (mask_wh.mask_w == true && mask_sf_load == true) {
        SM_load.sm_cos[sm_c] = pos_cs.cos_cache[d * WARP_Shfl + index.lane_id()];
        SM_load.sm_sin[sm_c] = pos_cs.sin_cache[d * WARP_Shfl + index.lane_id()];
    }

    //qk折中加载 WARP_SIZE_Shfl从起始和中间加载
    int rot_dim_offset = (mask_sf_load == true)? 
        d * WARP_Shfl + index.lane_id():
        pos_cs.cache_len + d * WARP_Shfl + (index.lane_id() - WARP_Shfl);

    if (first_h == true && mask_wh.mask_w == true) {
    SM_load.sm_k[sm_c] = key[
        pos_cs.seq_len_offset * k_seq_stride + 
        0 * k_head_stride +  
        rot_dim_offset
    ];
    }    

    SM_load.sm_q[sm_r][sm_c] = query[
        pos_cs.seq_len_offset * q_seq_stride +
        (h * WARP_NUMS + index.warp_id()) * q_head_stride +
        rot_dim_offset
    ];
    
    //等待片段加载
    SM_load.block_sync();

    //neox风格排列三角缓存
    if (mask_wh.mask_w == true && mask_sf_load == true) {
        SM_load.sm_cos[sm_c + WARP_Shfl] = SM_load.sm_cos[sm_c];
        SM_load.sm_sin[sm_c + WARP_Shfl] = SM_load.sm_sin[sm_c];
    }

    //neox风格旋转
    if (first_h == true && mask_wh.mask_w == true) {
        half k_ = SM_load.sm_k[sm_c];        
        k_ = (mask_sf_load == true)? k_:__hneg(k_);
        k_ = __shfl_xor_sync(MASK, k_, WARP_Shfl);
        SM_load.sm_k_rot[sm_c] = k_;
    }

    half q_ = SM_load.sm_q[sm_r][sm_c];        
    q_ = (mask_sf_load == true)? q_:__hneg(q_);
    q_ = __shfl_xor_sync(MASK, q_, WARP_Shfl);
    SM_load.sm_q_rot[sm_r][sm_c] = q_;

    //等待neox风格调整
    SM_load.block_sync();

    //计算旋转
    half k_cos, k_sin {__int2half_rn(0)};
    half q_cos, q_sin {__int2half_rn(0)};

    if (first_h == true && mask_wh.mask_w == true) {
        k_cos = __hmul_rn(SM_load.sm_k[sm_c], SM_load.sm_cos[sm_c]);
        k_sin = __hmul_rn(SM_load.sm_k_rot[sm_c], SM_load.sm_sin[sm_c]);
    }
    q_cos = __hmul_rn(SM_load.sm_q[sm_r][sm_c], SM_load.sm_cos[sm_c]);
    q_sin = __hmul_rn(SM_load.sm_q_rot[sm_r][sm_c], SM_load.sm_sin[sm_c]);

    //覆盖回qk张量 折中写回
    if (first_h == true && mask_wh.mask_w == true) {
    key[
        pos_cs.seq_len_offset * k_seq_stride + 
        0 * k_head_stride +  
        rot_dim_offset
    ] = __hadd_rn(k_cos, k_sin);
    }
    
    query[
        pos_cs.seq_len_offset * q_seq_stride +
        (h * WARP_NUMS + index.warp_id()) * q_head_stride +
        rot_dim_offset
    ] = __hadd_rn(q_cos, q_sin);
    }
    }
}

extern "C" __host__ void neox_style::rope_withyarn_neox_v1_launch(
    half* cos_sin_cache, int* positions, half* query, half* key,     
    int q_seq_stride, int q_head_stride,  
    int k_seq_stride, int k_head_stride,     
    int seq_len, int num_heads, int rotary_dim
) noexcept
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    dim3 block(BLOCK_X, BLOCK_Y);
    int grid_x = seq_len;
    int grid_y = 1;
    dim3 grid(grid_x, grid_y);

    using namespace neox_style;
    void* args[] = {
        &cos_sin_cache, &positions, &query, &key,   
        &q_seq_stride, &q_head_stride, 
        &k_seq_stride, &k_head_stride, 
        &seq_len, &num_heads, &rotary_dim
    };
    CUDA_CHECK(cudaLaunchKernel(
        (void*)&rope_withyarn_neox_v1<BLOCK_X, BLOCK_Y, BLOCK_Y*BLOCK_X>,
        grid, block,
        args, 0, cudaStreamDefault
    ));
    CUDA_CHECK(cudaGetLastError());
    return;
}
