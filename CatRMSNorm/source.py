import importlib
import importlib.util
import os
from typing import Optional, Union, Tuple
import torch
from torch import nn
import time

def load_cumla_decode_so(name: str, file: str, file_path: str) -> importlib.machinery.ModuleSpec:
    current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
    so_abs_path = os.path.join(current_dir, file)
    # 加载so文件
    spec = importlib.util.spec_from_file_location(
        name=name,
        location=so_abs_path
    )
    # 创建模块对象
    so_model = importlib.util.module_from_spec(spec)
    # 执行模块加载
    spec.loader.exec_module(so_model)
    print(f"SO文件加载成功")
    return so_model

class RMSNorm(nn.Module):
    """
    均方根层归一化（RMSNorm）
    核心逻辑：仅基于特征维度的均方根归一化，无均值中心化，相比LayerNorm更高效

    参数:
        dim (int): 输入张量的最后一维特征维度（如512/1536/7168）
        eps (float): 数值稳定性小常数，避免分母为0，默认1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim          # 目标特征维度
        self.eps = eps          # 数值稳定项
        # 可学习的缩放参数（初始化为1，此处设为不参与梯度更新）
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(
        self,
        x: torch.Tensor,        # 输入向量 512 1536 7168
        residual: Optional[torch.Tensor] = None,  # 残差张量，形状需与x一致
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播：RMSNorm核心计算，支持残差相加

        参数:
            x (torch.Tensor): 输入张量，最后一维必须等于初始化的dim
            residual (Optional[torch.Tensor]): 可选残差张量，用于残差连接

        返回:
            torch.Tensor / Tuple: 归一化后的张量；若传入residual则返回 (归一化结果, 残差中间值)
        """
        # 保存原始数据类型（避免混合精度下精度丢失）
        orig_dtype = x.dtype  # 示例：torch.float16/torch.bfloat16
        # 转换为float32计算（保证归一化数值稳定性）
        x = x.to(torch.float32)  # 形状不变：[*, D]
        # 残差相加（若传入residual）
        if residual is not None:
            # 残差张量转换为float32后与x相加，形状仍为[*, D]
            x = x + residual.to(torch.float32)
            # 保存相加后的残差（转回原始类型），用于后续返回
            residual = x.to(orig_dtype)  # 形状：[*, D]

        # 校验特征维度（防止输入维度不匹配）
        hidden_size = x.shape[-1]  # 获取最后一维维度：D
        if hidden_size != self.dim:
            raise ValueError(f"输入特征维度需为{self.dim}，但实际为{hidden_size}")

        # 计算均方根（RMS）核心逻辑
        # 计算每个样本在特征维度的平方值，形状：[*, D]
        # 对最后一维求均值 得到每个样本的平方均值，形状：[*, 1]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # 归一化：x / sqrt(平方均值 + eps)，rsqrt等价于1/sqrt，计算更高效，形状：[*, D]
        x = x * torch.rsqrt(variance + self.eps)
        # 转回原始数据类型（恢复混合精度）
        x = x.to(orig_dtype)  # 形状：[*, D]
        # 应用缩放参数（权重），形状：[*, D]
        x = x * self.weight

        # 返回结果：无残差则返回归一化张量，有残差则返回 (归一化张量, 残差中间值)
        if residual is None:
            return x  # 形状：[*, D]
        else:
            return x, residual  # x形状[*, D]，residual形状[*, D]

if __name__ == "__main__":
    dtype = torch.float16
    test_dim = 7168
    test_module = RMSNorm(test_dim).to(device="cuda", dtype=dtype)
    test_x = torch.randn(test_dim, dtype=dtype, device="cuda")
    test_res = torch.randn(test_dim, dtype=dtype, device="cuda")
    cu_test_x = test_x.clone()
    cu_test_res = test_res.clone()

    st = time.time()
    out_x, out_res = test_module(test_x, test_res)
    ed = time.time()
    print(f"计时 {(ed - st)*1000} ms")

    cu_module = load_cumla_decode_so("cu_catrmsnorm", "cu_Catrmsnorm_complie_v2.so", "./lib")
    cu_out_x, cu_out_res = cu_module.catrmsnorm(
        x=cu_test_x, residual=cu_test_res,
        dim=test_dim, eps=test_module.eps, weights=test_module.weight
    )

    for n in range(test_dim):
        print("偏移=", n)
        #print(f"python_x: {out_x[n]} cuda_x:{cu_out_x[n]}")
        #print(f"python_res: {out_res[n]} cuda_res:{cu_out_res[n]}")
        print(f"div_x:{out_x[n] - cu_out_x[n]} div_res:{out_res[n] - cu_out_res[n]}")
