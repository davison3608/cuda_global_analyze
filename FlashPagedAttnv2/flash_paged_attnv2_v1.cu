#include "cukernel.h"

template<32, 16, 1, 512, 16, 1>
__global__ __launch_bounds__(512)  void decode::flash_paged_attnv2_v1(
    float* __restrict__ Query,      //[1, 1, num_q_heads, qk_head_dim]
    float* __restrict__ K_cache,    //[N_blocks, BLOCK_N, num_kv_heads, qk_head_dim]
    float* __restrict__ V_cache,    //[N_blocks, BLOCK_N, num_kv_heads, kv_lora_rank] 
    int* __restrict__ block_table,  //[seq_lens // BLOCK_N]
    int num_q_heads,                //query头数目
    int num_kv_head,                //MQA下始终为1
    int qk_head_dim,                //qk低维长度 
    int kv_lora_rank,               //v低维长度
    int seq_lens, int splitk,       //有效tokens长度与分片数目
    float sm_scale,                 //缩放参数
    float logits,                   //tanh裁剪
    float* __restrict__ attn_out    //[1, splitk, num_q_heads, 2 + kv_lora_rank]
) 
{
    static constexpr int BLOCK_X = 32;
    static constexpr int BLOCK_Y = 16;
    static constexpr int BLOCK_Z = 1;
    constexpr int BLOCK_SZ = BLOCK_Y * BLOCK_X;

    #ifdef __CUDACC__
        constexpr int WARP_SZ = 32;
    #elifdef __HIP_PLATFORM_AMD__
        constexpr int WARP_SZ = 64;
    #else
        constexpr int WARP_SZ = 32;
    #endif
    constexpr int WARP_Sf = WARP_SZ / 2;
    constexpr int WARP_NS = BLOCK_Y;
    constexpr int MASK = 0xffffffff;
    
    constexpr int BLOCK_N = WARP_NS * 1;
    constexpr int BLOCK_H = 1;

    struct alignas(4) layout {
    public:
    __device__ layout() noexcept = default;
        int blo_x, blo_y, blo_z = -1;
        int blk_x, blk_y = -1;

    __device__ __forceinline__ void init() noexcept {
    this->blo_x = threadIdx.x;this->blo_y = threadIdx.y;this->blo_z = 0;
    this->blk_x = blockIdx.x;this->blk_y = blockIdx.y;
    }
    __device__ __forceinline__ int&warp_id() noexcept
    { return this->blo_y; }
    __device__ __forceinline__ int&lane_id() noexcept 
    { return this->blo_x; }
    __device__ __forceinline__ int&group_head() noexcept
    { return this->blk_y; }
    __device__ __forceinline__ int&group_sequence() noexcept
    { return this->blk_x; }
    };
    layout index();
    index.init();

    //分页表定位缓存逻辑
    struct PagManger {
    private:
        int stride_k_pag, stride_k_dim = -1;
        int stride_v_pag, stride_v_dim = -1;
        float* cache_k {nullptr}; 
        float* cache_v {nullptr};

    public:
    __device__ PagManger(int&dim_k, int&dim_v) {
    this->stride_k_dim = dim_k;
    this->stride_k_pag = BLOCK_N * dim_k;
    this->stride_v_dim = dim_v;
    this->stride_v_pag = BLOCK_N * dim_v;
    }
    __device__ PagManger() noexcept = delete; 

    __device__ void positioning_k(float* K_cache, int global_token_id) noexcept {
    int global_table_id = global_token_id / BLOCK_N;
    int group_table_id = global_token_id - global_table_id * BLOCK_N;
    int offset_glo = global_table_id * this->stride_k_pag;
    int offset_grp = group_table_id * this->stride_k_dim;
    this->cache_k = K_cache + offset_glo + offset_grp;
    }
    __device__ void positioning_v(float* V_cache, int global_token_id) noexcept {
    int global_table_id = global_token_id / BLOCK_N;
    int group_table_id = global_token_id - global_table_id * BLOCK_N;
    int offset_glo = global_table_id * this->stride_v_pag;
    int offset_grp = group_table_id * this->stride_v_dim;
    this->cache_v = V_cache + offset_glo + offset_grp;
    }
    __device__ __forceinline__ float* get_cache_k() const noexcept
    { return this->cache_k; }
    __device__ __forceinline__ float* get_cache_v() const noexcept
    { return this->cache_v; }
    };  
    PagManger manger(qk_head_dim, kv_lora_rank);
    
    //每个block所处 tokens/head范围
    struct Scope {
    public:
    __device__ Scope() noexcept {}
        int2 head_scope = make_int2(-1, -1);    
        int2 token_scope = make_int2(-1, -1);
        int token_base_len = -1;
        int token_rem_len = -1;
        int token_real_len = -1;
    };
    Scope scope();

    if (index.group_sequence() >= splitk)
        return;
    
    //head起始与终点范围 左闭右开
    scope.head_scope.x = index.group_head() * BLOCK_H;
    scope.head_scope.y = scope.head_scope.x + BLOCK_H;
    scope.head_scope.y = min(scope.head_scope.y, num_q_heads);

    //token起始与终点范围 左闭右开
    scope.token_base_len = seq_lens / splitk;
    scope.token_rem_len = seq_lens % splitk;
    
    int&token_start = scope.token_scope.x;
    int&token_end = scope.token_scope.y;
    if (index.group_sequence() == (splitk - 1)) {
        bool is_rem = scope.token_rem_len == 0;
        token_start = (splitk - 1) * scope.token_base_len; 
        token_end = (is_rem == false)? 
            token_start + scope.token_rem_len:
            token_start + scope.token_base_len;
    }
    else {
        token_start = index.group_sequence() * scope.token_base_len;
        token_end = token_start + scope.token_base_len;
    }
    token_end = min(token_end, seq_lens);

    //每个block 序列 头 低维 迭代次数
    scope.token_real_len = token_end - token_start;
    const int iter_n = (scope.token_real_len + BLOCK_N - 1) / BLOCK_N;
    constexpr int iter_h = 1;

    const int iter_dk = qk_head_dim / WARP_SZ;
    const int iter_dv = kv_lora_rank / WARP_SZ;

    //所有缓存定义 分片维护online值定义
    struct SM_view {
    public:
    __device__ SM_view() noexcept {}
        float alignas(4) sm_q[BLOCK_H][WARP_SZ];
        float alignas(4) sm_k[BLOCK_N][WARP_SZ];
        float alignas(4) sm_v[BLOCK_N][WARP_SZ];
        
        float alignas(4) sm_p[BLOCK_H][BLOCK_N];
        
        constexpr int MaxDv = 640;
        float alignas(4) sm_emax[BLOCK_H];
        float alignas(4) sm_eexp[BLOCK_H];
        float alignas(4) sm_acc[BLOCK_H][MaxDv];
    
    __device__ void reset_qkv(layout&index) noexcept {
    if (index.warp_id() < BLOCK_N) {
        this->sm_k[index.warp_id()][index.lane_id()] = 0.0f;
        this->sm_v[index.warp_id()][index.lane_id()] = 0.0f;
        if (index.warp_id() < BLOCK_H)
            this->sm_q[index.warp_id()][index.lane_id()] = 0.0f;
    }
    else
        return;
    }
    __device__ void reset_p(layout&index) noexcept {
    if (index.lane_id() == 0) 
        this->sm_p[index.blo_z][index.warp_id()] = 0.0f;
    else
        return;
    }
    __device__ void reset_verb(layout&index) noexcept {
    if (index.warp_id() < BLOCK_H) {
        constexpr int iter_dv = MaxDv / WARP_SZ;
        #pragma unroll
        for (int i=0; i<iter_dv; i++)
            this->sm_acc[index.warp_id()][i * WARP_SZ + index.lane_id()] = 0.0f;
        if (index.lane_id() != 0)
            return;
        this->sm_eexp[index.warp_id()] = 0.0f;
        this->sm_emax[index.warp_id()] = -INFINITY;
    }
    else
        return;
    }
    __device__ __forceinline__ void syncblock() const 
    { return __syncthreads(); }
    __device__ __forceinline__ void syncwarp() const  
    { return __syncwarp(MASK); }   
    };
    __shared__ SM_view SM_load;

    __shared__ float SM_old_scale;
    __shared__ float SM_cur_scale;

    //初始化online值
    SM_load.reset_verb(index);
    SM_load.syncblock();
    
    #pragma unroll
    for (int i_h=0; i_h<iter_h; i_h++) {
    //对于query头偏移
    int&offset_head_grp = i_h * BLOCK_H + index.blo_z;
    int offset_head_glo = index.group_head() * 1 + offset_head_grp;

    #pragma unroll
    for (int i_n; i_n<iter_n; i_n++) {
    SM_load.reset_p(index);
    
    //对于seq_lens的token偏移
    int offset_token_grp = i_n * BLOCK_N + index.warp_id();
    int offset_token_glo = index.group_sequence() * scope.token_real_len + 
        offset_token_grp;
    bool is_effect = (offset_token_grp < scope.token_real_len)? true:false;

    //根据全局偏移定位分页token阶段
    manger.positioning_k(K_cache, offset_token_glo);
    manger.positioning_v(V_cache, offset_token_glo);
    float* K_pagtoken = manger.get_cache_k();
    float* V_pagtoken = manger.get_cache_v();

    //原始注意力矩阵阶段
    for (int dk=0; dk<iter_dk; dk++) {
        int offset_dk = dk * WARP_SZ + index.lane_id();
        SM_load.sm_k[index.warp_id()][index.lane_id()] = (is_effect == true)? 
            K_pagtoken[offset_dk]:-INFINITY;

        if (index.warp_id() == 0)
        SM_load.sm_q[index.blo_z][index.lane_id()] = Query[
            0 * num_q_heads * qk_head_dim + offset_head_glo * qk_head_dim +
            offset_dk
        ];

        SM_load.syncblock();

        float qk_t = __fmul_rn(
            SM_load.sm_k[index.warp_id()][index.lane_id()], 
            SM_load.sm_q[index.blo_z][index.lane_id()];
        );

        #pragma unroll
        for (int sflen=WARP_Sf; sflen>0; sflen/=2) 
            qk_t += __shfl_xor_sync(MASK, qk_t, sflen);

        if (index.lane_id() == 0) 
            SM_load.sm_p[index.blo_z][index.warp_id()] += qk_t;
        
        SM_load.syncblock();
    }

    //缩放与裁剪阶段 越界token保持-inf
    if (index.lane_id() == 0 && is_effect == true) {
        float src_val, tan_val = 0.0f;
        src_val = SM_load.sm_p[index.blo_z][index.warp_id()];

        src_val = __fmul_rn(src_val, sm_scale);
        tan_val = tanhf(__fdiv_rn(src_val, logits));
        src_val = __fmul_rn(tan_val, logits);

        SM_load.sm_p[index.blo_z][index.warp_id()] = src_val;
    }

    SM_load.syncblock();

    float p_val {-INFINITY};
    bool mask_n = (index.lane_id() < BLOCK_N)? false:true;
    //注意力矩阵online softmax阶段
    if (index.warp_id() < BLOCK_H) {
    p_val = (mask_n == false)? 
        SM_load.sm_p[index.warp_id()][index.lane_id()]:-INFINITY;
    }

    if (index.warp_id() < BLOCK_H) {
    float head_max = p_val;
    #pragma unroll
    for (int sflen=WARP_Sf; sflen>0; sflen/=2) {
        float or_max = __shfl_xor_sync(MASK, p_val, sflen);
        head_max = fmaxf(head_max, or_max);
    }
    head_max = __shfl_sync(MASK, head_max, 0);

    SM_load.syncwarp();

    float old_max {-INFINITY};
    float new_max {-INFINITY};
    float&loc_max = head_max;

    old_max = SM_load.sm_emax[index.warp_id()];
    new_max = fmaxf(old_max, loc_max);
    SM_load.sm_emax[index.warp_id()] = new_max;

    //获取旧状态与局部缩放因子
    SM_old_scale = expf(old_max - new_max);
    SM_cur_scale = expf(loc_max - new_max);

    SM_load.syncwarp();

    //注意这里无效token指数转为0.0f
    p_val = expf(p_val);
    
    float old_exp {-INFINITY};
    float loc_exp {-INFINITY};

    loc_exp = p_val;
    #pragma unroll
    for (int sflen=WARP_Sf; sflen>0; sflen/=2) 
        loc_exp += __shfl_xor_sync(MASK, sflen, loc_exp);
    old_exp = SM_load.sm_eexp[index.warp_id()];
    loc_exp = __shfl_sync(MASK, loc_exp, 0);

    SM_load.syncwarp();

    SM_load.sm_eexp[index.warp_id()] = __fmul_rn(old_exp, SM_old_scale) +
        __fmul_rn(loc_exp, SM_cur_scale);

    if (mask_n == false)
        SM_load.sm_p[index.warp_id()][index.lane_id()] = p_val;
    }

    SM_load.syncblock();
    
    //累加加权输出阶段
    for (int dv=0; dv<iter_dv; dv++) {
        int offset_dv = dv * WARP_SZ + index.lane_id();
        //sm_p[BLOCK_H, BLOCK_N] * sm_v[BLOCK_N, WARP_SZ] = acc_dv[BLOCK_H, WARP_SZ]
        SM_load.sm_v[index.warp_id()][index.lane_id()] = (is_effect == true)? 
            V_pagtoken[offset_dv]:-INFINITY;

        SM_load.syncblock();

        if (index.warp_id() < BLOCK_H) {
        float val_sum = 0.0f;
        
        #pragma unroll
        for (int i=0; i<BLOCK_N; i++) {
            bool _is_effect = ((i_n * BLOCK_N + i)  < scope.token_real_len)? true:false;
            if (_is_effect == false)
                break;
            float dv_t = __fmul_rn(
                SM_load.sm_p[index.warp_id()][index.lane_id()],
                SM_load.sm_v[i][index.lane_id()]
            )
            val_sum += dv_t;
        }

        SM_load.sm_acc[index.warp_id()][offset_dv] += val_sum;
        }

        SM_load.syncblock();
    }
    //iter_n down
    }
    }

    __shared__ float combine_emax;
    __shared__ float combine_eexp;
    if (index.warp_id() < BLOCK_H && index.lane_id() == 0) {
        combine_emax = SM_load.sm_emax[index.warp_id()];
        combine_eexp = SM_load.sm_eexp[index.warp_id()];
    }

    SM_load.syncblock();
    
    int stride_attn_splitk = num_q_heads * (2 + kv_lora_rank);
    int stride_attn_head = (2 + kv_lora_rank);
    //写入中间注意力输出阶段 附加combine合并依赖
    if (index.warp_id() < BLOCK_H) {
    for (int dv=0; dv<iter_dv; dv++) {
        int offset_dv = dv * WARP_SZ + index.lane_id();
        float acc_val = SM_load.sm_v[index.warp_id()][index.lane_id()];

        attn_out[
            index.group_sequence() * stride_attn_splitk +
            index.group_head() * stride_attn_head +
            2 + offset_dv
        ] = acc_val;
        attn_out[
            index.group_sequence() * stride_attn_splitk +
            index.group_head() * stride_attn_head + 0
        ] = combine_emax;
        attn_out[
            index.group_sequence() * stride_attn_splitk +
            index.group_head() * stride_attn_head + 1
        ] = combine_eexp;
    }
    }
}
