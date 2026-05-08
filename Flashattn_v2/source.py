import triton
import triton.language as tl
import dataclasses

@dataclasses.dataclass(frozen=True)
class AttentionConfig:
    batch_size = 1
    max_seq_len: int = 153600
    num_q_head: int = 128
    num_kv_head: int = 1
    qk_head_dim: int = 576
    kv_lora_rank: int = 512
    splitK: int = 8
    block_head = 8
    block_seq = 64
config = AttentionConfig()

@triton.jit
def triton_tanh(x):
    exp_pos = tl.exp(x)
    exp_neg = tl.exp(-x)
    return (exp_pos - exp_neg) / (exp_pos + exp_neg)

@triton.jit
def mla_decode_split_simple(
    Q,          # [q_head_num, qk_head_dim]
    K,          # [max_seq_len, kv_head_num, qk_head_dim]
    V,          # [max_seq_len, kv_head_num, kv_lora_rank]
    Attn_Out,   # [SPLIT_K, q_head_num, kv_lora_rank + 1] —— partial output + logsumexp
    seq_len,    # scalar: 当前有效序列长度 (<= max_seq_len)
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_dim: tl.constexpr,
    kv_lora_rank: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,   # 沿序列维度的分块大小（如 64）
    BLOCK_SIZE_H: tl.constexpr,   # 沿 head 维度的分块大小（如 4）决定了每个线程块内部循环的粒度
    SPLIT_K: tl.constexpr,        # 沿序列方向 split 成多少份（用于并行）有多少个并行线程块（program）参与计算
    logit_cap: tl.constexpr,
):
    cur_head_group = tl.program_id(0)  # head 分组 ID
    split_id = tl.program_id(1)        # split-K 的 chunk ID

    # heads: [BLOCK_SIZE_H], e.g., [0,1,2,3] for group 0
    heads = cur_head_group * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    mask_h = heads < q_head_num  # [BLOCK_SIZE_H]

    # GQA 映射：多个 Q heads 共享一个 KV head
    # kv_head: [BLOCK_SIZE_H], 所有值 = 0 (因为 kv_head_num=1)
    kv_head = heads // (q_head_num // kv_head_num)
    kv_head = tl.where(mask_h, kv_head, 0)

    # 计算当前 split 的序列范围
    kv_chunk_size = tl.cdiv(seq_len, SPLIT_K)  # scalar
    start_n = split_id * kv_chunk_size         # scalar
    end_n = tl.minimum(start_n + kv_chunk_size, seq_len)  # scalar

    # 初始化累加器
    e_max = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")  # [BLOCK_SIZE_H]
    e_sum = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)                 # [BLOCK_SIZE_H]
    acc = tl.zeros([BLOCK_SIZE_H, kv_lora_rank], dtype=tl.float32)     # [BLOCK_SIZE_H, kv_lora_rank]

    offs_dv = tl.arange(0, kv_lora_rank)      # [kv_lora_rank]
    mask_dv = offs_dv < kv_lora_rank          # [kv_lora_rank]

    # 如果当前 split 无效（比如 seq_len 很短，某些 split 为空）
    if end_n <= start_n:
        out_base = split_id * q_head_num * (kv_lora_rank + 1)  # scalar offset
        # out_ptrs: [BLOCK_SIZE_H, kv_lora_rank]
        out_ptrs = out_base + heads[:, None] * (kv_lora_rank + 1) + offs_dv[None, :]
        tl.store(Attn_Out + out_ptrs, 0.0, mask=mask_h[:, None] & mask_dv[None, :])

        # lse_ptrs: [BLOCK_SIZE_H] —— 指向每个 head 的 logsumexp 位置（第 kv_lora_rank 列）
        lse_ptrs = out_base + heads * (kv_lora_rank + 1) + kv_lora_rank
        tl.store(Attn_Out + lse_ptrs, float("-inf"), mask=mask_h)
        return

    # 沿序列分块计算
    for n in range(start_n, end_n, BLOCK_SIZE_N):
        offs_n = n + tl.arange(0, BLOCK_SIZE_N)  # [BLOCK_SIZE_N]
        mask_n = offs_n < end_n                  # [BLOCK_SIZE_N]

        # Load Q: shape = [BLOCK_SIZE_H, qk_head_dim]
        offs_q = heads[:, None] * qk_head_dim + tl.arange(0, qk_head_dim)[None, :]
        q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)

        # Load K: 将 K[offs_n, kv_head, :] 转置为 [qk_head_dim, BLOCK_SIZE_N]
        # offs_k: [qk_head_dim, BLOCK_SIZE_N]
        offs_k = (
            offs_n[None, :] * kv_head_num * qk_head_dim +   # seq offset
            kv_head[:, None] * qk_head_dim +                # kv_head offset
            tl.arange(0, qk_head_dim)[:, None]              # dim offset
        )
        k = tl.load(K + offs_k, mask=mask_n[None, :], other=0.0)  # [qk_head_dim, BLOCK_SIZE_N]

        # QK^T: [BLOCK_SIZE_H, qk_head_dim] × [qk_head_dim, BLOCK_SIZE_N] → [BLOCK_SIZE_H, BLOCK_SIZE_N]
        qk = tl.dot(q, k.to(q.dtype))

        # 缩放 & logit cap
        sm_scale = qk_head_dim ** -0.5
        qk *= sm_scale  # [BLOCK_SIZE_H, BLOCK_SIZE_N]
        if logit_cap > 0:
            qk = logit_cap * triton_tanh(qk / logit_cap)

        # Mask invalid positions: [BLOCK_SIZE_H, BLOCK_SIZE_N]
        qk = tl.where(mask_h[:, None] & mask_n[None, :], qk, float("-inf"))

        # Load V: shape = [BLOCK_SIZE_N, kv_lora_rank]
        # offs_v: [BLOCK_SIZE_N, kv_lora_rank]
        offs_v = (
            offs_n[:, None] * kv_head_num * kv_lora_rank +
            kv_head[None, :] * kv_lora_rank +
            offs_dv[None, :]
        )
        v = tl.load(V + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0)  # [BLOCK_SIZE_N, kv_lora_rank]

        # Online softmax update
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)           # [BLOCK_SIZE_H]
        re_scale = tl.exp(e_max - n_e_max)                   # [BLOCK_SIZE_H]
        p = tl.exp(qk - n_e_max[:, None])                    # [BLOCK_SIZE_H, BLOCK_SIZE_N]

        # acc: [BLOCK_SIZE_H, kv_lora_rank]
        acc *= re_scale[:, None]                             # broadcasting [BLOCK_SIZE_H, 1]
        acc += tl.dot(p.to(v.dtype), v)                      # [BLOCK_SIZE_H, BLOCK_SIZE_N] × [BLOCK_SIZE_N, kv_lora_rank]
        e_sum = e_sum * re_scale + tl.sum(p, 1)              # [BLOCK_SIZE_H]
        e_max = n_e_max                                      # [BLOCK_SIZE_H]

    # Finalize: store partial output and logsumexp
    out_base = split_id * q_head_num * (kv_lora_rank + 1)
    # out_ptrs: [BLOCK_SIZE_H, kv_lora_rank]
    out_ptrs = out_base + heads[:, None] * (kv_lora_rank + 1) + offs_dv[None, :]
    tl.store(Attn_Out + out_ptrs, acc, mask=mask_h[:, None] & mask_dv[None, :])

    # lse_ptrs: [BLOCK_SIZE_H] —— 每个 head 的 logsumexp 存在第 (kv_lora_rank) 个位置
    lse_ptrs = out_base + heads * (kv_lora_rank + 1) + kv_lora_rank
    tl.store(Attn_Out + lse_ptrs, e_max + tl.log(e_sum), mask=mask_h)

@triton.jit
def mla_decode_combine(
    Mid_O, # 输入splitK分块中间结果(batch_size, num_q_heads=128, splitK, 513)
    o, # 输出合并后的最终结果(batch_size, num_q_heads=128, 512)
    B_Seqlen, # 输入每个batch的实际序列长度
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    SPLIT_K: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    kv_lora_rank: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < kv_lora_rank

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + kv_lora_rank

    for split_kv_id in range(0, SPLIT_K):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, SPLIT_K)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
