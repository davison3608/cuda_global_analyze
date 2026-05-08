#include "cukernel.h"

template<>
__global__ __launch_bounds__(1024) void stage_decode::flash_attn_v2_decode_v2
    <32, 32, 1, 2>(
    const half* __restrict__ Q, //[1, 1, q_head_num, qk_head_dim]
    const half* __restrict__ K, //[1, max_seq_len, kv_head_num, qk_head_dim]
    const half* __restrict__ V, //[1, max_seq_len, kv_head_num, kv_lora_rank]
    half* Attn_out,   //[q_head_num, splitk, kv_lora_rank + 1]
    int curr_seq_len, //当前有效缓存长度
    int q_head_num,   //q总头数目
    int qk_head_dim,  //qk低维秩
    int kv_lora_rank, //v低维秩
    int splitk,       
    float sm_scale,
    float logit_cap
)
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 32;
    constexpr int BLOCK_Z = 1;
    constexpr int Vec_len = 2;
    constexpr int BLOCK_SIZE = BLOCK_Z * BLOCK_Y * BLOCK_X;
    constexpr int WARP_NUMS = BLOCK_Z * BLOCK_Y;
    constexpr int WARP_SIZE = BLOCK_X;

    constexpr int WARP_N = 4;
    constexpr int BLOCK_H = BLOCK_Z;
    constexpr int BLOCK_N = BLOCK_Y * WARP_N;

    constexpr int MASK = 0xffffffff;
    constexpr int Sf_len0 = WARP_SIZE / 2;
    constexpr int Sf_len1 = Sf_len0 / 2;
    constexpr int Sf_len2 = Sf_len1 / 2;
    constexpr int Sf_len3 = Sf_len2 / 2;
    constexpr int Sf_len4 = Sf_len3 / 2;

    int blo_x = threadIdx.x;
    int blo_y = threadIdx.y; //对应序列内token
    int blo_z = threadIdx.z; //对应组内头 始终为0

    int warp_id = blo_z * BLOCK_Y + blo_y;
    int lane_id = blo_x;
    int blk_x = blockIdx.x; //对应序列分片
    int blk_y = blockIdx.y; //对应头组

    //每个block处理的头范围 左闭右开
    int group_head_st = blk_y * BLOCK_H;
    int group_head_ed = group_head_st + BLOCK_H;

    //每个block处理的splitk序列长度 左闭右开
    int split_size = (curr_seq_len + splitk - 1) / splitk;

    int split_start = blk_x * split_size;
    int split_end = min(split_start + split_size, curr_seq_len);

    //对于多余block提前退出
    if (split_start >= split_end)
        return;

    //Q K片段加载 V片段加载
    __shared__ half alignas(4) sm_q_load_dk[BLOCK_H][WARP_SIZE * Vec_len];
    __shared__ half alignas(4) sm_k_load_dk[BLOCK_N][WARP_SIZE * Vec_len];
    __shared__ half alignas(4) sm_v_load_dv[BLOCK_N][WARP_SIZE * Vec_len];

    //每个分组全局最值与指数和
    __shared__ float sm_group_emax[BLOCK_H];
    __shared__ float sm_group_esum[BLOCK_H];

    //加权区域(Dv_max_randk >= kv_lora_rank) 加权区域的缩放参数(在每个BLOCK_N迭代更新)
    constexpr int Dv_max_randk = 640;
    __shared__ half alignas(4) sm_acc[BLOCK_H][Dv_max_randk]; 
    __shared__ float sm_group_re_scale[BLOCK_H];

    //首warp初始化全局维护与累加区域 
    if (blo_y < BLOCK_H) {
        sm_group_emax[blo_y] = -INFINITY;
        sm_group_esum[blo_y] = __int2float_rn(0);

        #pragma unroll
        for (int i=0; i<Dv_max_randk/WARP_SIZE; i++)
        sm_acc[blo_y][i * WARP_SIZE + blo_x] = __int2half_rn(0);
    }
    __syncthreads();

    //warp加载四个片段 splitk向量化长度 
    split_size = split_end - split_start; //真实splitk长度
    int split_vec = split_size / WARP_N;
    
    //有效block计算序列迭代次数
    int split_seq_iter = (split_size + BLOCK_N - 1) / BLOCK_N;

    //Q K V内秩向量化
    int qk_head_dim_vec = qk_head_dim / Vec_len;
    int kv_lora_rank_vec = kv_lora_rank / Vec_len;

    //Q K秩迭代次数 V秩迭代次数
    int d_k_iter = qk_head_dim / (WARP_SIZE * Vec_len);
    int d_v_iter = kv_lora_rank / (WARP_SIZE * Vec_len);

    for (int step=0; step<split_seq_iter; step++) {
    //Q注意力头偏移 
    int offset_head = blk_y * BLOCK_H + blo_z;

    //当前split内的基准 token偏移 全局token偏移
    int offset_token_l_base = step * BLOCK_N;
    offset_token_l_base += warp_id * WARP_N;

    int offset_token_g_base = split_start;
    offset_token_g_base += offset_token_l_base;

    //边界向量掩码
    bool mask_l_vec = false;
    mask_l_vec = (offset_token_l_base >= split_vec)? true:false;

    //临时qk注意力矩阵首warp重置 
    __shared__ alignas(16) float sm_qk[BLOCK_H][BLOCK_N];
    if ((blo_y * BLOCK_X + blo_x) < BLOCK_N)
        sm_qk[blo_z][blo_y * BLOCK_X + blo_x] = 0.0f;
    __syncthreads();

    const half2* Q_vec2 = reinterpret_cast<const half2*>(Q);
    const half2* K_vec2 = reinterpret_cast<const half2*>(K);
    const half2* V_vec2 = reinterpret_cast<const half2*>(V);

    //qk内层迭代 所有warp循环WARP_N个token
    #pragma unroll
    for (int wp_t=0; wp_t<WARP_N; wp_t++) {
    //每个warp循环持有WARP_N个toekn元素 共组成BLOCK_N局部向量
    float qk_i {0.0f};
    
    #pragma unroll
    for (int n=0; n<d_k_iter; n++) {
    //向量起始基准
    int offset_dk_base = n * WARP_SIZE + lane_id;

    //加载q注意力头片段
    int&sm_q_r = blo_z;
    int sm_q_c_base = lane_id * Vec_len;
    if (warp_id < BLOCK_H) {
        half2 q_src = Q_vec2[offset_head * qk_head_dim_vec + offset_dk_base];
        sm_q_load_dk[sm_q_r][sm_q_c_base + 0] = q_src.x;
        sm_q_load_dk[sm_q_r][sm_q_c_base + 1] = q_src.y;
    }
        
    //加载k序列片段
    int sm_k_r_base = warp_id * WARP_N;
    int sm_k_c_base = lane_id * Vec_len;
    half2 k_src = (mask_l_vec)? make_half2(__int2half_rn(0), __int2half_rn(0)):
        K_vec2[(offset_token_g_base + wp_t) * qk_head_dim_vec + offset_dk_base];
    sm_k_load_dk[sm_k_r_base + wp_t][sm_k_c_base + 0] = k_src.x;
    sm_k_load_dk[sm_k_r_base + wp_t][sm_k_c_base + 1] = k_src.y;

    __syncthreads();

    //每个warp计算[1, 32 * 2] * [BLOCK_N//1, 32 * 2] 累加到qk缓存
    float2 qk_i_t {0.0f, 0.0f};
    qk_i_t.x = __half2float(sm_q_load_dk[sm_q_r][sm_q_c_base + 0]) * 
        __half2float(sm_k_load_dk[warp_id * WARP_N + wp_t][sm_k_c_base + 0]);
    qk_i_t.y = __half2float(sm_k_load_dk[sm_q_r][sm_q_c_base + 1]) * 
        __half2float(sm_k_load_dk[warp_id * WARP_N + wp_t][sm_k_c_base + 1]);

    qk_i_t.x = __shfl_xor_sync(MASK, qk_i_t.x, Sf_len0);
    qk_i_t.x = __shfl_xor_sync(MASK, qk_i_t.x, Sf_len1);
    qk_i_t.x = __shfl_xor_sync(MASK, qk_i_t.x, Sf_len2);
    qk_i_t.x = __shfl_xor_sync(MASK, qk_i_t.x, Sf_len3);
    qk_i_t.x = __shfl_xor_sync(MASK, qk_i_t.x, Sf_len4);
    qk_i_t.x = __shfl_sync(MASK, qk_i_t.x, 0);

    qk_i_t.y = __shfl_xor_sync(MASK, qk_i_t.y, Sf_len0);
    qk_i_t.y = __shfl_xor_sync(MASK, qk_i_t.y, Sf_len1);
    qk_i_t.y = __shfl_xor_sync(MASK, qk_i_t.y, Sf_len2);
    qk_i_t.y = __shfl_xor_sync(MASK, qk_i_t.y, Sf_len3);
    qk_i_t.y = __shfl_xor_sync(MASK, qk_i_t.y, Sf_len4);
    qk_i_t.y = __shfl_sync(MASK, qk_i_t.y, 0);

    if (lane_id == 0)
        qk_i += qk_i_t.x + qk_i_t.y;

    __syncthreads();
    }

    //qk缩放 logitcap
    int&sm_qk_r = blo_z;
    int sm_qk_c_base = warp_id * WARP_N;
    if (lane_id == 0) {
        qk_i = __fmul_rn(qk_i, sm_scale);
        qk_i = __tanhf(__fdiv_rn(qk_i, logit_cap));
        qk_i = __fmul_rn(qk_i, logit_cap);
        
        qk_i = (mask_l_vec)? -INFINITY:qk_i;
        sm_qk[sm_qk_r][sm_qk_c_base + wp_t] = qk_i;
    }
    }

    //等待qk原始矩阵
    __syncthreads();

    //online softmax 以block z轴的有效warp进入 每个warp处理BLOCK_N片段
    constexpr int WARP_ITER = (BLOCK_N + WARP_SIZE - 1) / WARP_SIZE;
    bool mask_w = (blo_z < BLOCK_H && warp_id < WARP_ITER)? false:true;
    {
    __shared__ float n_e_max[BLOCK_H];  //多头最新局部最值
    __shared__ float h_e_max[BLOCK_H];  //多头局部最值
    __shared__ float re_scale[BLOCK_H]; //累加调整因子
    __shared__ float h_e_sum[BLOCK_H];  //局部指数和

    __shared__ float sm_h_e_max[BLOCK_H][WARP_ITER]; 
    __shared__ float sm_h_e_sum[BLOCK_H][WARP_ITER]; 
    //初始化在线参数
    if (mask_w == false && lane_id == 0) {
        n_e_max[blo_z] = -INFINITY;
        h_e_max[blo_z] = -INFINITY;
        re_scale[blo_z] = 0.0f;
        h_e_sum[blo_z] = 0.0f;

        sm_h_e_max[blo_z][warp_id] = -INFINITY;
        sm_h_e_sum[blo_z][warp_id] = 0.0f;
    }
    __syncthreads();

    //每个头的局部最值 由WARP_ITER个warp负责
    if (mask_w == false) {
    int offset_token = warp_id * WARP_SIZE + lane_id;
    bool mask = (offset_token >= BLOCK_N)? true:false;

    float token = (mask)? -INFINITY:sm_qk[blo_z][offset_token];
    float src = token;
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len0));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len1));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len2));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len3));
    src = fmax(src, __shfl_xor_sync(MASK, src, Sf_len4));
    src = __shfl_sync(MASK, src, 0);    

    if (lane_id == 0)
        sm_h_e_max[blo_z][warp_id] = src;
    }

    //等待warp片段最值
    __syncthreads();

    //每个头的所有warp局部最大值由第一个warp合并 同时计算每个头最新最大值与调整因子
    if (mask_w == false && warp_id == 0) {
    bool mask = (lane_id >= WARP_ITER)? true:false;

    float h_t = (mask)? -INFINITY:sm_h_e_max[blo_z][lane_id];
    h_t = fmax(h_t, __shfl_xor_sync(MASK, h_t, Sf_len0));
    h_t = fmax(h_t, __shfl_xor_sync(MASK, h_t, Sf_len1));
    h_t = fmax(h_t, __shfl_xor_sync(MASK, h_t, Sf_len2));
    h_t = fmax(h_t, __shfl_xor_sync(MASK, h_t, Sf_len3));
    h_t = fmax(h_t, __shfl_xor_sync(MASK, h_t, Sf_len4));
    h_t = __shfl_sync(MASK, h_t, 0);   

    if (lane_id == 0) {
        h_e_max[blo_z] = fmaxf(h_e_max[blo_z], h_t);
        //最新局部最值与本次大迭代调整因子
        n_e_max[blo_z] = fmaxf(h_e_max[blo_z], sm_group_emax[blo_z]);
        re_scale[blo_z] = __expf(sm_group_emax[blo_z] - n_e_max[blo_z]);
        //更新全局最大值
        sm_group_emax[blo_z] = n_e_max[blo_z];
    }
    }

    //同步每个头在线参数广播
    __syncthreads();

    //qk注意力矩阵数值稳定 由WARP_ITER个warp负责
    if (mask_w == false) {
    int offset_token = warp_id * WARP_SIZE + lane_id;
    bool mask = (offset_token >= BLOCK_N)? true:false;

    float token = (mask)? -INFINITY:sm_qk[blo_z][offset_token];
    token = __expf(token - n_e_max[blo_z]); //注意这一步将原始qk的掩码token变为0.0f 后续不影响指数和

    if (mask == false)
        sm_qk[blo_z][offset_token] = token;
    }

    //qk矩阵的每个头保证稳定后统一读取
    __syncthreads();

    //每个头的局部指数和计算 由WARP_ITER个warp负责 缩放旧指数和并累加
    if (mask_w == false) {
    int offset_token = warp_id * WARP_SIZE + lane_id;
    bool mask = (offset_token >= BLOCK_N)? true:false;

    float token = (mask)? 0.0f:sm_qk[blo_z][offset_token];
    float src = token;
    src += __shfl_xor_sync(MASK, src, Sf_len0);
    src += __shfl_xor_sync(MASK, src, Sf_len1);
    src += __shfl_xor_sync(MASK, src, Sf_len2);
    src += __shfl_xor_sync(MASK, src, Sf_len3);
    src += __shfl_xor_sync(MASK, src, Sf_len4);
    src = __shfl_sync(MASK, src, 0);    

    if (lane_id == 0)
        sm_h_e_sum[blo_z][warp_id] = src;
    }

    //等待warp片段指数和
    __syncthreads();

    //每个头的所有warp局部指数和由第一个warp合并 同时累加到全局指数和
    if (mask_w == false && warp_id == 0) {
    bool mask = (lane_id >= WARP_ITER)? true:false;

    float s_t = (mask)? 0.0f:sm_h_e_sum[blo_z][lane_id];
    s_t += __shfl_xor_sync(MASK, s_t, Sf_len0);
    s_t += __shfl_xor_sync(MASK, s_t, Sf_len1);
    s_t += __shfl_xor_sync(MASK, s_t, Sf_len2);
    s_t += __shfl_xor_sync(MASK, s_t, Sf_len3);
    s_t += __shfl_xor_sync(MASK, s_t, Sf_len4);
    s_t = __shfl_sync(MASK, s_t, 0);   

    if (lane_id == 0) {
        h_e_sum[blo_z] = s_t;
        //旧值缩放并累加
        float o_e_sum = sm_group_esum[blo_z];
        o_e_sum = o_e_sum * re_scale[blo_z] + h_e_sum[blo_z];
        //覆盖到全局
        sm_group_esum[blo_z] = o_e_sum;
    }
    }

    //重新覆盖本次序列片段的累加调整缩放因子
    if (mask_w == false && warp_id == 0 && lane_id == 0)
        sm_group_re_scale[blo_z] = re_scale[blo_z];    
    }

    //等待加权区域的调整因子
    __syncthreads();
    
    //v加权迭代 所有warp循环WARP_N个token
    for (int n=0; n<d_v_iter; n++) {
    //向量起始基准
    int offset_dv_base = n * WARP_SIZE + lane_id;
    
    for (int wp_t=0; wp_t<WARP_N; wp_t++) {
    //加载V片段
    int sm_v_r_base = warp_id * WARP_N;
    int sm_v_c_base = lane_id * Vec_len;
    half2 v_src = (mask_l_vec)? make_half2(__int2half_rn(0), __int2half_rn(0)):
            V_vec2[(offset_token_g_base + wp_t) * kv_lora_rank_vec + offset_dv_base];
    sm_v_load_dv[sm_v_r_base + wp_t][sm_v_c_base + 0] = v_src.x;
    sm_v_load_dv[sm_v_r_base + wp_t][sm_v_c_base + 1] = v_src.y;
    }

    __syncthreads();

    //qk与WARP_SIZE列的所有BLOCK_Y行乘积 首warp在BLOCK_N迭代
    if (warp_id < BLOCK_H) {
    //首warp循环持有BLOCK_N个累加值 共组成WARP_SIZE * 2局部向量
    float sum_pv_t[Vec_len] {0.0f};

    #pragma unroll
    for (int bn=0; bn<BLOCK_N; bn++) {
        sum_pv_t[0] += __fmul_rn(
            sm_qk[blo_z][bn],
            __half2float(sm_v_load_dv[bn][lane_id * Vec_len + 0])
        );
        sum_pv_t[1] += __fmul_rn(
            sm_qk[blo_z][bn],
            __half2float(sm_v_load_dv[bn][lane_id * Vec_len + 1])
        );
    }

    //写入到缓存本次迭代的对应列 每个线程处理2向量 同时调整缩放
    half2 o_acc = make_half2(__int2half_rn(0), __int2half_rn(0));
    o_acc.x = sm_acc[blo_z][offset_dv_base * Vec_len + 0];
    o_acc.y = sm_acc[blo_z][offset_dv_base * Vec_len + 1];

    o_acc.x = __hmul_rn(__float2half_rn(sm_group_re_scale[blo_z]), o_acc.x);
    o_acc.y = __hmul_rn(__float2half_rn(sm_group_re_scale[blo_z]), o_acc.y);
    o_acc.x = __hadd_rn(o_acc.x, __float2half_rn(sum_pv_t[0]));
    o_acc.y = __hadd_rn(o_acc.y, __float2half_rn(sum_pv_t[1]));

    sm_acc[blo_z][offset_dv_base * Vec_len + 0] = o_acc.x;
    sm_acc[blo_z][offset_dv_base * Vec_len + 1] = o_acc.y;
    }
    //本次内积不被下一次加载污染
    __syncthreads();
    }
    }

    //合并阶段附加参数 logsumexp
    float lse = sm_group_emax[blo_z] + __logf(sm_group_esum[blo_z]);

    //写回到中间注意力输出 [q_head_num, splitk, kv_lora_rank]
    if (warp_id < BLOCK_H) {
    //注意力头偏移 分片偏移
    int&offset_head = blk_y; //由于BLOCK_H固定为1 blk_y * BLOCK_H + blo_z
    int&offset_splitk = blk_x;
    
    #pragma unroll
    for (int n=0; n<d_v_iter; n++) {
        int offset_dv = n * WARP_SIZE + lane_id;

        Attn_out[
            offset_head * (splitk * (kv_lora_rank + 1)) +
            offset_splitk * (kv_lora_rank + 1) +
            offset_dv
        ] = sm_acc[warp_id][offset_dv];
    }

    Attn_out[
        offset_head * (splitk * (kv_lora_rank + 1)) +
        offset_splitk * (kv_lora_rank + 1) +
        kv_lora_rank
    ] = __float2half_rn(lse);
    }
}

extern "C" void stage_decode::v2_launch(
    half* Q, half* K, half* V, half* Attn_out,
    int curr_seq_len, int q_head_num, 
    int qk_head_dim, int kv_lora_rank,
    int splitk, float logit_cap
) noexcept
{
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 32;
    constexpr int BLOCK_Z = 1;
    constexpr int Vec_len = 2;
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


}
