#include "cukernel.h"

template<>
__global__ __launch_bounds__(256) void GroupedTopkfused_fp32_v1
    <32, 8, 256>(
    const float* __restrict__ gating_x,     //[N_tokens, n_routed_experts]
    const float* __restrict__ score_bias,   //[n_routed_experts]
    int N_tokens, int n_routed_experts,     //1 -- 256
    const int topk_num,                     //8
    const int num_groups,                   //8
    const int topk_groups,                  //4
    float route_scale,
    float* __restrict__ topk_weights,       //[N_tokens, topk_num]
    int* __restrict__ topk_ids              //[N_tokens, topk_num]
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP_NUMS = BLOCK_Y;
    constexpr int WARP_SIZE = BLOCK_X;
    constexpr int MASK = 0xFFFFFFFF;

    struct alignas(16) layout {
    __device__ layout() noexcept = default;

    public:
    int32_t blo_x {-1}; //对应n_routed_experts
    int32_t blo_y {-1}; //对应groups分组
    int32_t blk_x {-1}; //对应tokens数目 优先为0
    int32_t blk_y {-1}; //无
        
    __device__ void init() noexcept {
        this->blo_x = threadIdx.x;
        this->blo_y = threadIdx.y;
        this->blk_x = blockIdx.x; 
        this->blk_y = 0; 
    }
    __device__ __forceinline__ int blo_rank()
    { return this->blo_y * BLOCK_X + this->blo_x; }
    __device__ __forceinline__ int&warp_id() 
    { return this->blo_y; }
    __device__ __forceinline__ int&lane_id() 
    { return this->blo_x; }
    };
    layout index; 
    index.init();

    //sigmoid与偏置叠加阶段
    struct Score {
    __device__ Score() noexcept {}

    public:
    //sigmoid处理原始值
    float original_scores {-INFINITY};
    //叠加偏置的值
    float bias_scores {-INFINITY};
    //Nan则不参与sigmoid 后续筛选设置为-inf
    bool mask_Nan {false};
    //检查输入是否有Nan
    __device__ __forceinline__ void isNanandin() {        
        this->mask_Nan = __isnanf(this->original_scores) == 1;
        this->mask_Nan = __isnanf(this->bias_scores == 1)? true:this->mask_Nan;
    }
    //检测当前block持有数据是否掺杂Nan
    __device__ __forceinline__ int isNanAny_sync() {
        int isnan = this->mask_Nan? 1:0;
        return __syncthreads_count(isnan);
    }
    //sigmoid计算 同时添加偏置
    __device__ __forceinline__ void sigmoid(float&val, float&b) {
        float tmp = val;
        tmp = -tmp;
        tmp = __expf(tmp);
        tmp = __fadd_rn(tmp, 1.0f);
        tmp = __fdiv_rn(1.0f, tmp);
        val = tmp;
        b = __fadd_rn(b, val);
    }
    //提交任务 忽略Nan
    __device__ void commit() noexcept
    { this->sigmoid(this->original_scores, this->bias_scores); }
    };
    
    //scores集合
    Score scores;
    if (index.blk_x >= N_tokens)
        return;

    //分组专家数目 256/8 = 32
    int group_experts = n_routed_experts / num_groups; 
    int&token_offset = index.blk_x; 
    int&group_offset = index.lane_id(); 
    
    //每个warp负责的分组是否超出专家数目
    bool mask_group = false;
    mask_group = (group_offset >= group_experts)? true:false;
    if (mask_group == false) {
    float val = gating_x[
        token_offset * n_routed_experts + 
        index.warp_id() * group_experts + 
        group_offset
    ];
    float b = score_bias[
        index.warp_id() * group_experts + 
        group_offset
    ];

    scores.original_scores = val;
    scores.bias_scores = b;
    }
    
    scores.isNanandin();
    scores.commit();

    //groups topk阶段
    struct TSwap {
    __device__ TSwap() noexcept = default;
    
    public:
    //总是选择更大的值 若相同则选择索引更小的
    __device__ __forceinline__ void thread_swap (
        float&val, int&id, float&or_val, int&or_id
    ) {
        bool mask_swap = (or_val > val || (or_val == val && or_id < id));
        val = mask_swap? or_val:val;
        id = mask_swap? or_id:id;
    }
    //副本权重分数交换行为跟随着偏置偏置权重分数
    __device__ __forceinline__ void thread_swap (
        float&val, int&id, float&or_val, int&or_id, float&fval, float&or_fval
    ) {
        bool mask_swap = (or_val > val || (or_val == val && or_id < id));
        val = mask_swap? or_val:val;
        id = mask_swap? or_id:id;
        fval = mask_swap? or_fval:fval;
    }
    __device__ __forceinline__ void thread_swap (
        float&val, float&or_val
    ) { val = (or_val >= val)? or_val:val; }
    };
    TSwap tswap;

    constexpr int MAX_TOPK_NUMS = 16; //WARP_NUMS >= num_groups 16 >= topk_num
    constexpr int MAX_TOPK_GROUPS = 8; //实际为4

    __shared__ alignas(4) float SM_group_max[WARP_NUMS]; //WARP_NUMS >= num_groups
    __shared__ alignas(4) int SM_top_groups[MAX_TOPK_GROUPS]; //8 >= topk_groups

    __shared__ alignas(4) float SM_group_weights[WARP_NUMS * MAX_TOPK_NUMS]; 
    __shared__ alignas(4) float SM_group_weights_gather[WARP_NUMS * MAX_TOPK_NUMS];
    __shared__ alignas(4) int SM_group_ids[WARP_NUMS * MAX_TOPK_NUMS]; 

    __shared__ alignas(4) float SM_sin_group_weights_bias[MAX_TOPK_NUMS];
    __shared__ alignas(4) float SM_sin_group_weights[MAX_TOPK_NUMS];
    __shared__ alignas(4) int SM_sin_group_ids[MAX_TOPK_NUMS];

    struct SM_view {
    __device__ SM_view() noexcept {}

    public:
    float* sm_grp_max = nullptr; //[WARP_NUMS]
    int* sm_top_grp = nullptr;   //[MAX_TOPK_GROUPS]
    float* sm_grp_w = nullptr;   //[WARP_NUMS * MAX_TOPK_NUMS]
    float* sm_grp_w_g = nullptr; //[WARP_NUMS * MAX_TOPK_NUMS]
    int* sm_grp_i = nullptr;     //[WARP_NUMS * MAX_TOPK_NUMS]
    float* sm_sin_wb = nullptr;  //[MAX_TOPK_NUMS] 
    float* sm_sin_w = nullptr;   //[MAX_TOPK_NUMS]
    int* sm_sin_i = nullptr;     //[MAX_TOPK_NUMS]

    __device__ void SM_bind() noexcept {
    this->sm_grp_max = SM_group_max;
    this->sm_top_grp = SM_top_groups;
    this->sm_grp_w = SM_group_weights;
    this->sm_grp_w_g = SM_group_weights_gather;
    this->sm_grp_i = SM_group_ids;
    this->sm_sin_wb = SM_sin_group_weights_bias;
    this->sm_sin_w = SM_sin_group_weights;
    this->sm_sin_i = SM_sin_group_ids;
    }
    //初始化SM数据
    __device__ __forceinline__ void reset(layout&index) {
    if (index.lane_id() >= MAX_TOPK_NUMS)
        return;
    if (index.lane_id() == 0) {
        this->sm_grp_max[index.warp_id()] = -INFINITY;
        if (index.warp_id() < MAX_TOPK_GROUPS)
        this->sm_top_grp[index.warp_id()] = -1;
    }
    this->sm_grp_w[index.warp_id() * MAX_TOPK_NUMS + index.lane_id()] = -INFINITY;
    this->sm_grp_w_g[index.warp_id() * MAX_TOPK_NUMS + index.lane_id()] = -INFINITY;
    this->sm_grp_i[index.warp_id() * MAX_TOPK_NUMS + index.lane_id()] = -1;
    if (index.warp_id() == 0) {
        this->sm_sin_wb[index.lane_id()] = -INFINITY;
        this->sm_sin_w[index.lane_id()] = -INFINITY;
        this->sm_sin_i[index.lane_id()] = -1;
    }
    }
    __device__ __forceinline__ void block_sync() const
    { __syncthreads(); }
    __device__ __forceinline__ void warp_sync() const
    { __syncwarp(MASK); }
    };
    
    SM_view SM_load;
    SM_load.SM_bind();
    SM_load.reset(index);
    SM_load.block_sync();

    constexpr int WARP_Shfl = WARP_SIZE / 2;
    constexpr int Shfl_1 = WARP_Shfl / 2;
    constexpr int Shfl_2 = Shfl_1 / 2;
    constexpr int Shfl_3 = Shfl_2 / 2;
    constexpr int Shfl_4 = Shfl_3 / 2;

    float val_scores {-INFINITY};
    float val_original {-INFINITY}; 
    int idx_original {-1};

    //分组专家内 topk(2).sum(dim=-1)
    {
    if (mask_group == false) {
        val_scores = scores.original_scores;
        val_original = scores.bias_scores;
        idx_original = index.warp_id() * group_experts + group_offset;
    }

    float&val = val_original; 
    int&idx = idx_original;
    float *sm_maxs = SM_load.sm_grp_max; 

    float top1 = -INFINITY;
    float top2 = -INFINITY;
    int top1_idx = -1;
    int top2_idx = -1;
    
    float2 top_w2 = make_float2(top1, top2);
    int2 top_i2 = make_int2(top1_idx, top2_idx);

    float current_val = val;  
    int current_idx = idx;

    //warp内所有元素参与比较 找到第1大值
    #pragma unroll
    for (int offset=WARP_Shfl; offset>0; offset/=2) {
        float other_val = __shfl_xor_sync(MASK, current_val, offset);
        int other_idx = __shfl_xor_sync(MASK, current_idx, offset);
        tswap.thread_swap(
            current_val, current_idx, 
            other_val, other_idx
        );
    }
    top1 = __shfl_sync(MASK, current_val, 0);  
    top_w2.x = top1;
    top_i2.x = current_idx;
    
    //将等于第1大值的元素标记为无效
    current_val = (val == top1 && idx == current_idx)? -INFINITY:val; 

    SM_load.warp_sync();
    current_idx = idx;

    #pragma unroll
    for (int offset=WARP_SIZE/2; offset>0; offset/=2) {
        float other_val = __shfl_xor_sync(MASK, current_val, offset);
        int other_idx = __shfl_xor_sync(MASK, current_idx, offset);
        tswap.thread_swap(
            current_val, current_idx,
            other_val, other_idx
        );
    }
    top2 = __shfl_sync(MASK, current_val, 0);  
    top_w2.y = top2;
    top_i2.y = current_idx;

    //将本次缓存写入到共享内存 写入1次
    if (index.lane_id() == 0) 
        sm_maxs[index.warp_id()] = __fadd_rn(top_w2.x, top_w2.y); 
    }
    
    //等待每个专家最值有效
    SM_load.block_sync();

    //所有warp局部专家分组最值交由首warp处理
    if (index.warp_id() == 0) {
    float group_max {-INFINITY};
    int group_idx {-1};
    float *sm_maxs = SM_load.sm_grp_max; 
    int* sm_grps = SM_load.sm_top_grp;

    if (index.lane_id() < num_groups) {
        group_max = sm_maxs[index.lane_id()];
        group_idx = index.lane_id();
    }

    #pragma unroll
    for (int i=0; i<topk_groups; i++) {
    float max_val = group_max;
    int max_idx = group_idx;

    #pragma unroll
    for (int offset_shfl=WARP_Shfl; offset_shfl>0; offset_shfl /= 2) {
        float or_max_val = __shfl_xor_sync(MASK, max_val, offset_shfl);
        int or_max_idx = __shfl_xor_sync(MASK, max_idx, offset_shfl);
        tswap.thread_swap(
            max_val, max_idx, 
            or_max_val, or_max_idx
        );
    }

    //等待线程交换
    SM_load.warp_sync();

    //所有线程持有当前局部最大值 冗余广播保证稳定
    max_val = __shfl_sync(MASK, max_val, 0);
    max_idx = __shfl_sync(MASK, max_idx, 0);

    //写入当前第i个最值到共享内存 写入1次
    if (index.lane_id() == 0) 
        sm_grps[i] = max_idx;

    //能匹配刚刚写入的第k最值的线程 不再参与后续候选
    if (group_max == max_val && group_idx == max_idx) {
        group_max = -INFINITY;
        group_idx = -1;
    }

    //等待本次写入
    SM_load.warp_sync();
    }
    }

    //等待topk(4)分组专家有效
    SM_load.block_sync();

    //true表示该分组的专家屏蔽
    bool mask_inf {true}; 
    if (index.lane_id() == 0) {
    #pragma unroll
    for (int i=0; i<topk_groups; i++) {
    int group_idx {-1};
    group_idx = SM_load.sm_top_grp[i];

    //一旦发现则解除屏蔽
    if (index.warp_id() == group_idx) 
    { mask_inf = false; break; }
    }
    }
    //从0线程广播
    mask_inf = __shfl_sync(MASK, mask_inf, 0);

    //每个warp处理局部topk(8) 256/8 = 32
    if (mask_inf == false) {
    float thread_val = val_original;
    float thread_gather = val_scores;
    int thread_idx = idx_original;

    float *sm_vals = SM_load.sm_grp_w;
    float* sm_gathers = SM_load.sm_grp_w_g;
    int *sm_ids = SM_load.sm_grp_i;

    #pragma unroll
    for (int k=0; k<topk_num; k++) {
    float max_val = thread_val;
    float max_fval = thread_gather;
    int max_idx = thread_idx;
    
    #pragma unroll
    for (int offset_shfl=WARP_Shfl; offset_shfl>0; offset_shfl /= 2) {
        float or_max_val = __shfl_xor_sync(MASK, max_val, offset_shfl);
        float or_max_fval = __shfl_xor_sync(MASK, max_fval, offset_shfl);
        int or_max_idx = __shfl_xor_sync(MASK, max_idx, offset_shfl);
        tswap.thread_swap(
            max_val, max_idx, 
            or_max_val, or_max_idx,
            max_fval, or_max_fval
        );
    }

    //等待线程交换
    SM_load.warp_sync();

    //所有线程持有当前局部最大值 冗余广播保证稳定
    max_val = __shfl_sync(MASK, max_val, 0);
    max_fval = __shfl_sync(MASK, max_fval, 0);
    max_idx = __shfl_sync(MASK, max_idx, 0);

    if (index.lane_id() == 0) {
        sm_vals[index.warp_id() * MAX_TOPK_NUMS + k] = max_val;
        sm_gathers[index.warp_id() * MAX_TOPK_NUMS + k] = max_fval;
        sm_ids[index.warp_id() * MAX_TOPK_NUMS + k] = max_idx;
    }

    //能匹配刚刚写入的第k最值的线程 不再参与后续候选
    if (thread_val == max_val && thread_idx == max_idx) {
        thread_val = -INFINITY;
        thread_idx = -1;
    }

    //等待本次写入
    SM_load.warp_sync();
    }
    }

    //等待局部topk(8)有效
    SM_load.block_sync();

    //单token总topk长度 8*16=128 每个线程处理小向量长度 128/32=4 128/64=2
    constexpr int Topk_len = WARP_NUMS * MAX_TOPK_NUMS; 
    constexpr int Vec_len = Topk_len / WARP_SIZE; 

    //所有warp局部topk(8)交由首warp合并
    if (index.warp_id() == 0) {
    float val_len[Vec_len] {-INFINITY};
    float fval_len[Vec_len] {-INFINITY};
    int idx_len[Vec_len] {-1};

    float *sm_vals = SM_load.sm_grp_w;
    float* sm_gathers = SM_load.sm_grp_w_g;
    int *sm_ids = SM_load.sm_grp_i;

    float* sm_wb = SM_load.sm_sin_wb;
    float* sm_w = SM_load.sm_sin_w;
    int* sm_i = SM_load.sm_sin_i;

    //加载每个线程小向量
    #pragma unroll
    for (int l=0; l<Vec_len; l++) {
        val_len[l] = sm_vals[index.lane_id() * Vec_len + l];
        fval_len[l] = sm_gathers[index.lane_id() * Vec_len + l];
        idx_len[l] = sm_ids[index.lane_id() * Vec_len + l];
    }

    #pragma unroll
    for (int k=0; k<topk_num; k++) {
    float thread_max = -INFINITY;
    float thread_fmax = -INFINITY;
    int thread_idx = -1;
    
    //根据偏置后分数来选举小向量最值
    #pragma unroll
    for (int t=0; t<Vec_len; t++) {
        bool ismax = val_len[t] > thread_max;
        thread_max = ismax? val_len[t]:thread_max;
        thread_fmax = ismax? fval_len[t]:thread_fmax;
        thread_idx = ismax? idx_len[t]:thread_idx;
    }

    #pragma unroll
    for (int offset_shfl=WARP_Shfl; offset_shfl>0; offset_shfl /= 2) {
        float or_max = __shfl_xor_sync(MASK, thread_max, offset_shfl);
        float or_fmax = __shfl_xor_sync(MASK, thread_fmax, offset_shfl);
        int or_idx = __shfl_xor_sync(MASK, thread_idx, offset_shfl);
        tswap.thread_swap(
            thread_max, thread_idx, 
            or_max, or_idx,
            thread_fmax, or_fmax
        );
    }

    //等待线程交换
    SM_load.warp_sync();

    //所有线程持有当前局部最大值 冗余广播保证稳定
    thread_max = __shfl_sync(MASK, thread_max, 0);
    thread_fmax = __shfl_sync(MASK, thread_fmax, 0);
    thread_idx = __shfl_sync(MASK, thread_idx, 0);

    if (index.lane_id() == 0) {
        sm_wb[k] = thread_max;
        sm_w[k] = thread_fmax;
        sm_i[k] = thread_idx;
    }

    //移除已选中的元素 可能有多个相同的值 但索引应该唯一
    #pragma unroll
    for (int t=0; t<Vec_len; t++) {
        bool mask_inf = (val_len[t] == thread_max && idx_len[t] == thread_idx);
        val_len[t] = mask_inf? -INFINITY:val_len[t];
        idx_len[t] = mask_inf? -1:idx_len[t];
    }

    //等待本次写入
    SM_load.warp_sync();
    }    
    }

    //等待全局合并有效
    SM_load.block_sync();

    __shared__ alignas(2) float SM_weights_all;
    float _weights {0.0f};
    float _fweights {0.0f};

    bool mask_toplen = false;
    bool is_scale = false;

    //权重结果缩放阶段
    if (index.warp_id() == 0) {
    mask_toplen = index.lane_id() < topk_num;
    is_scale = (route_scale != 1.0f)? true:false;
    
    float* top_results = SM_load.sm_sin_w;
    int* top_idx = SM_load.sm_sin_i;

    _weights = mask_toplen? top_results[index.lane_id()]:_weights;
    _weights = __fadd_rn(_weights, __shfl_xor_sync(MASK, _weights, WARP_Shfl));
    _weights = __fadd_rn(_weights, __shfl_xor_sync(MASK, _weights, Shfl_1));
    _weights = __fadd_rn(_weights, __shfl_xor_sync(MASK, _weights, Shfl_2));
    _weights = __fadd_rn(_weights, __shfl_xor_sync(MASK, _weights, Shfl_3));
    _weights = __fadd_rn(_weights, __shfl_xor_sync(MASK, _weights, Shfl_4));
    
    if (index.lane_id() == 0)
        SM_weights_all = _weights;
    }

    //等待缩放因子有效
    int warmup = scores.isNanAny_sync();
    _weights = 0.0f;

    //缩放同时应用routed scale
    if (index.warp_id() == 0) {
    float* top_results = SM_load.sm_sin_w;
    int* top_idx = SM_load.sm_sin_i;

    _weights = mask_toplen? top_results[index.lane_id()]:_weights;
    _weights = mask_toplen? __fdiv_rn(_weights, SM_weights_all):_weights;
    _fweights = (mask_toplen == true && is_scale == true)? 
        __fmul_rn(_weights, route_scale):_weights;

    //写回有效权重分数与索引
    if (index.lane_id() < topk_num) {
        int&token_offset = index.blk_x; 
        int&top_offset = index.lane_id(); 
    if (warmup == 0) {
        topk_weights[
            token_offset * topk_num + 
            top_offset
        ] = _fweights;
        topk_ids[
            token_offset * topk_num + 
            top_offset
        ] = top_idx[index.lane_id()];
    }
    else 
        topk_ids[
            token_offset * topk_num + 
            top_offset
        ] = index.lane_id();
    }
    }
}