from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm

# mla
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        scale_fmt (Optional[str]): Format for quantization scale.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Optional[str] = None
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, scale_fmt: Optional[str] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size, scale_fmt)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16
    scale_fmt: Optional[str] = None

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias, self.scale_fmt)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    MoE 模型的门控机制模块（核心路由组件）
    功能：为每个输入 Token 计算所有专家的匹配分数，筛选 Top-K 个专家并输出权重，支持分组路由优化
    """
    def __init__(self, args: ModelArgs):
        """
        初始化门控层，定义路由所需的参数和可学习权重

        Args:
            args (ModelArgs): 模型配置参数，需包含：
                - dim: 输入特征维度（与 MoE 模块输入维度一致）
                - n_activated_experts: 每个 Token 激活的专家数（即 topk）
                - n_routed_experts: 全局稀疏专家总数
                - n_expert_groups: 专家分组数（用于分组路由，减少计算量）
                - n_limited_groups: 每个 Token 路由到的组数量（分组路由的 Top-K 组）
                - score_func: 分数归一化函数（'softmax' 或 'sigmoid'）
                - route_scale: 路由权重的缩放因子（平衡专家贡献）
        """
        super().__init__()
        # 基础路由参数
        # 输入特征维度（例：4096，与 MoE 模块的 dim 一致）
        self.dim = args.dim
        # 每个 Token 最终激活的专家数 当前为6
        self.topk = args.n_activated_experts
        # 专家分组数当前1组
        self.n_groups = args.n_expert_groups
        # 每个 Token 先筛选的组数量
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func  # 分数归一化方式（控制权重分布）
        self.route_scale = args.route_scale  # 权重缩放因子（例：1.0，可调整专家贡献强度）

        # 门控核心权重（线性层参数，用于计算输入与每个专家的匹配分数）
        # 形状：[n_routed_experts, dim] → 对应线性层 W，输入 x 与 W^T 相乘得到分数
        # 计算逻辑：scores = x @ weight.T → [N, dim] @ [dim, n_routed_experts] = [N, n_routed_experts]
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))

        # 可选偏置项（仅当输入维度为 7168 时添加）
        # 形状：[n_routed_experts] → 为每个专家的分数添加独立偏置
        self.bias = nn.Parameter(
            torch.empty(args.n_routed_experts, dtype=torch.float32)
        ) if self.dim >= 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：计算输入与专家的匹配分数 → （可选分组路由）→ 筛选 Top-K 专家 → 输出权重和索引

        Args:
            x (Tensor): 输入张量（来自 MoE 模块的展平输入），形状 [N, dim]（N = batch×seq_len）
        Returns:
            Tuple[Tensor, Tensor]:
                - weights: 每个 Token 对 Top-K 专家的归一化权重，形状 [N, topk]
                - indices: 每个 Token 选中的 Top-K 专家的全局索引，形状 [N, topk]
        """
        # 计算原始匹配分数（输入 × 门控权重，得到每个 Token 对所有专家的原始分数）
        # 线性变换：[N, dim] → [N, n_routed_experts]，每个元素代表Token与该专家的匹配度
        st = time.time()
        scores = linear(x, self.weight)  # 等价于 x @ self.weight.T（若 linear 是标准线性层）

        # 分数归一化（按配置的函数转换，使分数符合概率分布或合理范围）
        if self.score_func == "softmax":
            # Softmax归一化：每个Token的所有专家分数求和为1，突出高匹配度专家
            scores = scores.softmax(dim=-1, dtype=torch.float32)  # 形状保持 [N, n_routed_experts]
        else:
            # Sigmoid归一化：每个专家分数映射到 [0,1]，可保留多个中等匹配度专家
            scores = scores.sigmoid()  # 形状保持 [N, n_routed_experts]

        # 保存归一化后的原始分数（后续用于提取Top-K专家的权重，避免偏置干扰）
        original_scores = scores

        # 添加偏置项（仅当bias存在时，为每个专家的分数加偏置）
        if self.bias is not None:
            scores = scores + self.bias

        # 分组路由优化（减少专家筛选范围，提升效率）
        if self.n_groups > 1: # 671B = 8
            scores = scores.view(x.size(0), self.n_groups, -1) # [N, n_routed_experts] [N, n_groups, n_experts_per_group]
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1) # [N, 8]
            indices = group_scores.topk(self.topk_groups, dim=-1)[1] # [N, 4] 最高概率的四个组
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1) # [N, n_groups, n_experts_per_group]中四个组被屏蔽

        # 筛选Top-K专家：从所有（或选中组的）专家中，选择分数最高的topk个
        # indices：选中专家的全局索引，形状 [N, topk]
        indices = torch.topk(scores, self.topk, dim=-1)[1]

        # 提取 Top-K 专家的权重：从原始归一化分数中，按索引提取对应专家的权重
        # 形状变化：original_scores [N, n_routed_experts] → 按indices [N, topk] 提取 → [N, topk]
        # 表示N个token的每个单词对激活的topk个专家的匹配分数
        weights = original_scores.gather(1, indices)

        # Sigmoid 模式下的二次归一化：确保Top-K专家权重求和为1（与 softmax 保持一致的权重逻辑）
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)  # 形状保持 [N, topk]

        # 权重缩放：通过 route_scale 调整权重强度（可控制专家贡献的整体幅度）
        weights *= self.route_scale
        # 该 Token 对「第 k 个激活专家」的归一化匹配分数（权重），反映专家对该Token的贡献程度
        # 该 Token 第 k 个激活专家的全局索引
        ed = time.time()
        print(f"Gate门控网络计时 {(ed - st) * 1000} MS")
        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        初始化单个稀疏专家网络，定义网络层权重与维度映射关系

        Args:
            dim (int): 输入和输出特征维度（需与 MoE 模块输入维度一致，例：4096）
            inter_dim (int): 中间隐藏层维度（通常为输入维度的 2~4 倍，例：16384）
        """
        super().__init__()
        # 普通线性层（无分布式并行），输入→中间层（维度扩张）
        self.w1 = Linear(dim, inter_dim)
        # 普通线性层（无分布式并行），中间层→输出（维度收缩，对齐输入）
        self.w2 = Linear(inter_dim, dim)
        # 普通线性层（无分布式并行），与w1并行生成门控权重（SwiGLU激活用）
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：基于 SwiGLU 激活的门控式特征提取

        Args:
            x (Tensor): 输入张量（来自 MoE 模块的路由筛选），形状为 [K, dim]
                        其中 K 是当前专家被门控层选中的 Token 总数（批量×序列中匹配的 Token）
        Returns:
            Tensor: 专家计算后的输出张量，形状为 [K, dim]（与输入维度一致，支持 MoE 模块加权累加）
        """
        # SwiGLU激活逻辑：SiLU(w1(x)) * w3(x) → 门控式特征提取
        # 注意这里的F.silu(self.w1(x))处理后和w3(x)的两个[K, inter_dim]形状的矩阵为逐元素相乘并非矩阵乘
        gated_hidden = F.silu(self.w1(x)) * self.w3(x)
        return self.w2(gated_hidden)


class MoE(nn.Module):
    """
    混合专家（Mixture-of-Experts, MoE）模块
    采用「稀疏路由专家 + 共享专家」架构，适配分布式训练场景
    """
    def __init__(self, args: ModelArgs):
        """
        初始化MoE模块，核心是划分分布式专家、构建门控与专家网络

        Args:
            args (ModelArgs): 模型配置参数，需包含：
                - dim: 输入/输出特征维度
                - n_routed_experts: 全局稀疏路由专家总数
                - n_activated_experts: 每个输入Token激活的专家数
                - moe_inter_dim: 专家网络中间层维度
                - n_shared_experts: 共享专家的数量（用于构建共享MLP）
        """
        super().__init__()
        # 1. 基础维度参数
        self.dim = args.dim  # 输入特征维度（例：hidden_size=4096）
        # 2. 分布式专家划分参数（确保全局专家数能被进程数整除）
        assert args.n_routed_experts % world_size == 0, f"全局专家数必须能被进程数整除（当前进程数={world_size}）"
        # 全局稀疏专家总数
        self.n_routed_experts = args.n_routed_experts
        # 当前进程负责的本地专家数
        self.n_local_experts = args.n_routed_experts // world_size
        # 当前进程专家的起始索引 由于进程数为1 则rank为0 当前进程从0到n_local_experts个专家
        self.experts_start_idx = rank * self.n_local_experts
        # 当前进程专家的结束索引
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts

        # 每个Token激活的专家数 目前为6
        self.n_activated_experts = args.n_activated_experts
        # 3. 门控层（核心路由组件） 输入：[N, dim] 输出：权重[ N, n_activated_experts ]、索引[ N, n_activated_experts ]
        self.gate = Gate(args)

        # 4. 稀疏路由专家列表（仅初始化当前进程负责的专家）
        self.experts = nn.ModuleList()
        for expert_idx in range(self.n_routed_experts):
            # 仅对当前进程负责的专家索引，实例化Expert网络
            if self.experts_start_idx <= expert_idx < self.experts_end_idx:
                # Expert 输入形状：[K, dim]（K为路由到该专家的Token数）
                # Expert 输出形状：[K, dim]（与输入维度一致，便于加权累加）
                self.experts.append(Expert(args.dim, args.moe_inter_dim))
            else:
                # 非当前进程负责的专家，设为None（节省显存）
                self.experts.append(None)

        # 5. 共享专家网络（所有Token都会经过，提升特征关联性）
        # MLP输入：[N, dim]，输出：[N, dim]（通过n_shared_experts控制中间层容量）
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：门控路由 → 本地专家计算 → 共享专家计算 → 结果融合

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]（批量×序列长度×特征维度）
        Returns:
            torch.Tensor: 输出张量，形状与输入一致 [batch_size, seq_len, dim]
        """
        # 1. 展平输入：将[batch, seq_len, dim]转为[N, dim]（N = batch×seq_len，方便批量计算）
        # 保存原始形状，用于最终恢复
        origin_shape = x.size()
        # 形状变化：[B, S, D] → [B×S, D]（B=batch，S=seq_len，D=dim） 此时N代表所有token
        x_flat = x.view(-1, self.dim)

        # 2. 门控路由：生成每个Token的专家选择权重和索引
        # weights：每个Token对选中专家的权重，形状 [N, n_activated_experts]
        # indices：每个Token选中的专家全局索引，形状 [N, n_activated_experts]
        weights, indices = self.gate(x_flat)

        # 3. 初始化稀疏专家输出张量（与输入展平后形状一致，初始值为0）
        routed_output = torch.zeros_like(x_flat)  # 形状 [N, D]

        # 4. 统计每个专家被激活的次数（用于跳过无输入的专家，优化效率）
        # 展平indices后统计频次，形状为 [n_routed_experts]（每个元素是对应专家的激活次数）
        expert_counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()

        # 5. 遍历当前进程负责的所有专家，计算对应Token的输出
        for local_expert_idx in range(self.n_local_experts):
            # 计算当前专家的全局索引（本地索引→全局索引映射）
            global_expert_idx = self.experts_start_idx + local_expert_idx
            # 跳过无Token激活的专家（避免无效计算）
            if expert_counts[global_expert_idx] == 0:
                continue

            # 获取当前专家实例（非None，因为是当前进程负责的专家）
            current_expert = self.experts[global_expert_idx]

            # 筛选出路由到当前专家的Token索引：
            # idx：被路由到当前专家的Token在x_flat中的索引（形状 [K,]，K为激活次数）
            # top_k_idx：该Token在其激活专家列表中的位置（0或1，因n_activated_experts通常为2）
            # 形状上idx代表indices的行，top_i_idx代表indices的列
            idx, top_k_idx = torch.where(indices == global_expert_idx)

            # 专家计算：当前专家处理选中的Token，并乘以对应权重（加权融合）
            # 形状变化：x_flat[idx] → [K, D] → 专家输出 [K, D] → 乘权重 [K, 1] → 结果 [K, D]
            expert_out = current_expert(x_flat[idx]) * weights[idx, top_k_idx, None]

            # 累加当前专家的输出到总稀疏输出中（同一Token可能激活多个专家，需叠加）
            routed_output[idx] += expert_out

        # 6. 共享专家计算：所有Token都经过共享专家，提取基础特征
        shared_output = self.shared_experts(x_flat)  # 形状 [N, D]

        # 7. 分布式通信：汇总所有进程的稀疏专家输出（多卡场景）
        # 因不同进程负责不同专家，需通过all_reduce将所有专家的输出汇总到每个进程
        if world_size > 1:
            dist.all_reduce(routed_output, op=dist.ReduceOp.SUM)  # 所有进程的routed_output求和

        # 8. 结果融合与形状恢复：稀疏专家输出 + 共享专家输出，再恢复为原始输入形状
        final_output = routed_output + shared_output  # 形状 [N, D]
        return final_output.view(origin_shape)  # 形状变化：[B×S, D] → [B, S, D]

"""
1，专家权重延迟加载 仅在经过门控后需要计算时才加载到显存 并且计算完毕的中间激活及时释放
2，共享专家与稀疏专家并行
3，只处理被激活的专家（提前去重） 按激活专家分组，聚合所有 Token（同一专家的 Token 合并为一个批量）
4，分布式场景下，对于稀疏专家拆分到多个组对应不同设备计算，对于MLP层的共享专家进行张量拆分计算
"""
