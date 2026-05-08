#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import logging
import random
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import ctypes
from pathlib import Path
logger = logging.getLogger(__name__)

def rotate_neox(x: torch.Tensor) -> torch.Tensor:
    """
    NeoX 风格的向量旋转操作（用于 LLaMA、DeepSeek Indexer 等）。
    将输入向量分为前后两半 (x1, x2)，然后返回 (-x2, x1)，
    相当于在二维平面上对每一对维度进行 90 度逆时针旋转。

    示例：[x0, x1, x2, x3] → [-x2, -x3, x0, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)
def rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    """
    GPT-J 风格的向量旋转操作（用于 GPT-J、DeepSeek MLA 等）。
    将输入向量按奇偶维度拆分：偶数位为 x1，奇数位为 x2，
    然后对每一对 (x_{2i}, x_{2i+1}) 构造旋转：(-x_{2i+1}, x_{2i})，
    最后展平回原始维度顺序。

    示例：[x0, x1, x2, x3] → [-x1, x0, -x3, x2]
    """
    x1 = x[..., ::2]  # 偶数索引：x0, x2, ...
    x2 = x[..., 1::2]  # 奇数索引：x1, x3, ...
    x = torch.stack((-x2, x1), dim=-1)  # 构造 [(-x1, x0), (-x3, x2), ...]
    return x.flatten(-2)  # 展平最后两维 → [-x1, x0, -x3, x2, ...]

# YaRN工具函数
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    """
    根据期望的旋转次数（num_rotations），反推在总维度 `dim` 中，
    对应的频率维度位置（即哪些维度会在 max_position 内完成指定圈数的旋转）。
    用于确定 YaRN 中需要“修正”的高频/低频边界。
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    """
    计算 YaRN 中需要混合插值与外推的维度范围 [low, high]。
    - low_rot：高频端（如 beta_fast=32）→ 对应较小维度索引（更敏感）
    - high_rot：低频端（如 beta_slow=1）→ 对应较大维度索引
    返回 clamp 后的有效维度区间。
    """
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)
def yarn_linear_ramp_mask(
    low: float,
    high: float,
    dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    生成一个线性斜坡掩码（ramp mask），用于在 [low, high] 区间内
    平滑过渡插值（interpolation）和外推（extrapolation）策略。
    - 在 low 以下：完全使用外推
    - 在 high 以上：完全使用插值
    - 中间：线性混合
    防止 low == high 导致除零错误。
    """
    if low == high:
        high += 0.001  # 避免除零
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)  # 限制在 [0, 1]
    return ramp_func
def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    """
    计算 YaRN 的幅度缩放因子（magnitude scaling），
    用于补偿因上下文扩展导致的 attention logits 幅度过大问题。
    - 当 scale <= 1（无扩展）：mscale = 1.0
    - 当 scale > 1：mscale ≈ 0.1 * log(scale) + 1，随扩展倍数缓慢增长
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(nn.Module):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        reference: bool = False,
    ) -> None:
        super().__init__()
        assert dtype == torch.float16, ("FH DEBUG: Assertion: dtype should be torch.float16", dtype)

        self.scaling_factor = scaling_factor        # 缩放因子 用于拉伸最长上下文长度
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.reference = reference
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        # super().__init__(
        #     head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        # )

        self.head_size = head_size      # 注意力头维度大小
        self.rotary_dim = rotary_dim    # 实际参与编码的维度大小 通常等于head_size
        self.max_position_embeddings = max_position_embeddings # 支持的最大上下文长度（即最大 token 数）
        self.base = base                # RoPE 中频率基底（base of the frequency exponent），控制位置编码的频率分布
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache() # 预计算缓存cos_sin_cache
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor      # [self.max_position_embeddings * self.scaling_factor, self.rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        """
        计算 YaRN 混合频率：低频用插值（压缩频率），高频用外推（保持原频率），
        中间用线性斜坡平滑过渡。
        返回：形状为 (rotary_dim // 2,) 的逆频率向量 inv_freq
        """
        pos_freqs = self.base ** (
            torch.arange(
                0,
                self.rotary_dim,
                2,
                dtype=torch.float,
                # device=current_platform.device_type,
            )
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """
        预计算所有位置（0 到 max_pos * scaling_factor）的 cos 和 sin 值，
        并乘以幅度缩放因子 mscale。
        返回：形状为 (max_pos * scaling, rotary_dim) 的缓存张量，
            前半为 cos，后半为 sin（通过 chunk 分离）
        """
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            # device=current_platform.device_type,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.forward_torch(positions, query, key, offsets)

    def forward_torch(
        self,                                    # decode阶段seqlen=1
        positions: torch.Tensor,                 # [batch_size, seq_len=1] 每个token在序列中的绝对位置索引
        query: torch.Tensor,                     # [batch_size, seq_len=1, num_heads=32, head_size]
        key: Optional[torch.Tensor] = None,      # [batch_size, seq_len=1, num_heads=1, head_size]
        offsets: Optional[torch.Tensor] = None,  # 通常为 None（当前实现不支持）
    ):
        global query_pass, key_pass
        assert key is not None
        assert offsets is None, "FH DEBUG: Assertion: offsets should be None"

        if self.cos_sin_cache.device != positions.device:
            self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)

        # [batch_size, seqlen=1, num_head_mla, rope_dim_mla]
        query_rot = query[..., :self.rotary_dim]
        # [batch_size, seqlen=1, 1, rope_dim_mla]
        key_rot = key[..., :self.rotary_dim]

        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]

        # 提取pos绝对位置的cossin缓存[batchsize=1, seqlen, rotary_dim]
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        # cos前32列[batch_size, seq_len, rotary_dim//2] sin后32列[batch_size, seq_len, rotary_dim//2]
        cos, sin = cos_sin.chunk(2, dim=-1)

        # Indexer风格
        if self.is_neox_style:
            cos = cos.repeat(1, 1, 2).unsqueeze(-2) # [batch_size, seq_len, 1, rotary_dim]
            sin = sin.repeat(1, 1, 2).unsqueeze(-2) # 对rotary_dim//2维度整体复制两次 a,b -- a,b,a,b
        # MLA风格
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2) # [batch_size, seq_len, 1, rotary_dim]
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2) # 对rotary_dim//2维度逐元素复制两次 a,b -- a,a,b,b

        # 应用旋转 Indexer风格或者MLA风格
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
        # [batch_size, num_head_mla, rope_dim_mla] * [batch_size, seq_len, 1, rotary_dim] 对32个头的向量逐元素乘积
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        # [batch_size, 1, rope_dim_mla] * [batch_size, seq_len, 1, rotary_dim] 对单个头的向量逐元素乘积
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    def forward_hip(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ):
        bs, seq, heads, dim = query.shape
        assert dim == self.rotary_dim
        assert (heads % 16 == 0) and (dim % 64 == 0) and (bs == 1)
        assert query.dtype == torch.float16 and key.dtype == torch.float16
        assert positions.is_contiguous()

        _, q_seq_stride, q_head_stride, _ = query.stride()
        _, k_seq_stride, k_head_stride, _ = key.stride()

        so_dir = Path(__file__).parent
        if self.is_neox_style:
            lib_path = so_dir / "neox_rope.so"
            func_name = "rope_withyarn_neox_v1_launch"
        else:
            lib_path = so_dir / "gptj_rope.so"
            func_name = "rope_withyarn_gptj_v1_launch"

        lib = ctypes.CDLL(str(lib_path))
        func = getattr(lib, func_name)
        func.argtypes = [
            ctypes.c_void_p,  # half* cos_sin_cache
            ctypes.c_void_p,  # int* positions
            ctypes.c_void_p,  # half* query
            ctypes.c_void_p,  # half* key
            ctypes.c_int,     # q seq stride
            ctypes.c_int,     # q head stride
            ctypes.c_int,     # k seq stride
            ctypes.c_int,     # k head stride
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # num_heads
            ctypes.c_int      # rotary_dim
        ]
        func.restype = None  # void

        func(
            self.cos_sin_cache.data_ptr(),
            positions.data_ptr(),
            query.data_ptr(),
            key.data_ptr() if key is not None else 0,
            q_seq_stride, q_head_stride,
            k_seq_stride, k_head_stride,
            seq,
            heads,
            dim
        )
        return query, key

