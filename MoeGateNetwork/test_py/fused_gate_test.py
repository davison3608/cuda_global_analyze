#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import logging
import ctypes
from pathlib import Path
import torch
import torch.nn as nn
"""logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)"""


# https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566
def grouped_topk_deepseek(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "sigmoid",
    e_score_correction_bias: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
    use_sorted: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1, dtype=torch.float32)
    else:
        scores = gating_output.sigmoid()
    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias

    if num_expert_group > 1:
        scores = scores.view(hidden_states.size(0), num_expert_group, -1)
        if e_score_correction_bias is None:
            group_scores = scores.amax(dim=-1)
        else:
            group_scores = scores.topk(2, dim=-1, sorted=use_sorted)[0].sum(dim=-1)
        indices = group_scores.topk(topk_group, dim=-1, sorted=use_sorted)[1]
        mask = scores.new_ones(hidden_states.size(0), num_expert_group, dtype=bool).scatter_(1, indices, False)
        scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

    indices = torch.topk(scores, topk, dim=-1, sorted=use_sorted)[1]
    weights = original_scores.gather(1, indices)

    if renormalize:
        weights /= weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        weights *= routed_scaling_factor

    return weights, indices


# https://github.com/vllm-project/vllm/blob/327a02d8db86e57f2488779b0c1b133da5f03fb5/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py#L167
class GroupedTopk(nn.Module):
    """GroupedTopk used by the Deepseek-V2 and Deepseek-V3 model."""

    def __init__(
        self,
        topk: int,
        renormalize: bool,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        num_fused_shared_experts: int = 0,
        use_sorted: bool = True,
    ) -> None:
        super().__init__()
        self.topk = topk # 8
        self.renormalize = renormalize
        self.num_expert_group = num_expert_group # 8
        self.topk_group = topk_group # 4
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.num_fused_shared_experts = num_fused_shared_experts
        self.use_sorted = use_sorted

    def forward(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
       return self.forward_torch(hidden_states, gating_output, e_score_correction_bias)

    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ):
        global original_scores
        assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

        if self.scoring_func == "softmax":
            scores = torch.softmax(gating_output, dim=-1)
        elif self.scoring_func == "sigmoid":
            # scores[1, 256], scores.dtype=torch.float16
            scores = gating_output.sigmoid()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        num_token = scores.size(0)
        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_scores = scores
            # [1, 256]half + [256,]float
            scores = scores + e_score_correction_bias.unsqueeze(0)
            # group_scores[1, 8], group_scores.dtype=torch.float32
            group_scores = scores.view(num_token, self.num_expert_group, -1).topk(2, dim=-1).values.sum(dim=-1)
        else:
            group_scores = scores.view(num_token, self.num_expert_group, -1).max(dim=-1).values # [n, n_group] [1, 8]

        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=self.use_sorted).indices # [n, top_k_group] [1, 4]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group] [1, 8]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group] [1, 8]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_token, self.num_expert_group, scores.size(-1) // self.num_expert_group)
            .reshape(num_token, -1)
        )  # [n, e] [1, 256]
        tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e] [1, 256]

        if e_score_correction_bias is not None:
            topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1, sorted=self.use_sorted).indices
            # Use original unbiased scores for the routing weights
            # topk_weights.shape=torch.Size([1, 8]), topk_weights.dtype=torch.float16
            topk_weights = original_scores.gather(1, topk_ids)  # [n, topk] [1, 8]
        else:
            topk_weights, topk_ids = torch.topk(tmp_scores, k=self.topk, dim=-1, sorted=self.use_sorted)

        if self.renormalize:
            # [1, 8] / [1, 1] dtype=torch.float16
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        if self.routed_scaling_factor != 1.0:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ):
        num_token = gating_output.size(0)
        num_experts = gating_output.size(1)
        is_float16 = True if gating_output.dtype is torch.float16 else False

        dev = gating_output.device
        topk_values = torch.full((num_token, self.topk), float('nan'), dtype=torch.float32, device=dev)
        topk_indices = torch.zeros((num_token, self.topk), dtype=torch.int32, device=dev)

        so_dir = Path(__file__).parent
        lib_path = so_dir / "GroupedTopkfused.so"
        func_name = "GroupedTopkfused_v1_launch"

        lib = ctypes.CDLL(str(lib_path))
        func = getattr(lib, func_name)
        func.argtypes = [
            ctypes.c_void_p,  # void* gating_x
            ctypes.c_void_p,  # float* score_bias
            ctypes.c_int,     # N_tokens
            ctypes.c_int,     # n_routed_experts
            ctypes.c_int,     # topk_num
            ctypes.c_int,     # num_groups
            ctypes.c_int,     # topk_groups
            ctypes.c_float,   # route_scale
            ctypes.c_void_p,  # topk_weights
            ctypes.c_void_p,  # topk_ids
            ctypes.c_bool     # is_float16
        ]
        func.restype = None  # void

        func(
            gating_output.data_ptr(),
            e_score_correction_bias.data_ptr(),
            num_token, num_experts,
            self.topk, self.num_expert_group, self.topk_group,
            self.routed_scaling_factor,
            topk_values.data_ptr(),
            topk_indices.data_ptr(),
            is_float16
        )
        return topk_values, topk_indices


def grouped_topk_test():
    num_token = 1
    hidden_size = 7168 # config.hidden_size      # 7168
    num_experts = 256 # config.n_routed_experts # 256
    topk = 8 # config.num_experts_per_tok     # 8
    num_expert_group = 8  # config.n_group     # 8
    topk_group = 4 # config.topk_group        # 4
    scoring_func = "sigmoid" # config.scoring_func    # "sigmoid"
    routed_scaling_factor = 1.0

    layer = GroupedTopk(
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        num_fused_shared_experts=0,
    )

    for dtype in (torch.float16, torch.float32):
        hidden_states = torch.randn((num_token, hidden_size), dtype=dtype, device="cuda")
        gating_output = torch.randn((num_token, num_experts), dtype=dtype, device="cuda")
        e_score_correction_bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")

        # 注意检查 nan 的情况
        nan_ratio = 0.1
        with torch.no_grad():
            # gating_output
            mask = torch.rand_like(gating_output) < nan_ratio
            gating_output[mask] = float('nan')
            # e_score_correction_bias
            mask = torch.rand_like(e_score_correction_bias) < nan_ratio
            e_score_correction_bias[mask] = float('nan')
        # torch and hip
        topk_weights_torch, topk_ids_torch = layer.forward_torch(
            hidden_states, gating_output,
            e_score_correction_bias
        )
        assert torch.all((0 <= topk_ids_torch) & (topk_ids_torch < num_experts))
        topk_weights_c, topk_ids_c = layer.forward_hip(
            hidden_states, gating_output,
            e_score_correction_bias
        )
        torch.cuda.synchronize()

        if torch.allclose(topk_weights_torch, topk_weights_c, atol=1e-3) is False:
            diff = (topk_weights_torch - topk_weights_c).abs()
            max_err = diff.max().item()
            print(f"weights error\nMax absolute: {max_err:.8f}")
            raise ValueError
        if torch.allclose(topk_ids_torch, topk_ids_c, atol=0) is False:
            print(f"ids error")
            raise ValueError

        # deepseek realize
        topk_weights_ds, topk_ids_ds = grouped_topk_deepseek(
            hidden_states,
            gating_output,
            topk=topk,
            renormalize=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )
        torch.testing.assert_close(topk_weights_ds.type_as(topk_weights_torch), topk_weights_torch, atol=1e-2, rtol=1e-2, equal_nan=True)
        torch.testing.assert_close(topk_ids_ds.type_as(topk_ids_torch), topk_ids_torch, atol=0, rtol=0) # int 类型，应该无任何偏差

        print(f"{dtype} test pass")

def main():
    torch.set_default_dtype(torch.float16)
    torch.set_default_device("cuda")
    for n in range(500):
        print(f"iter={n}")
        grouped_topk_test()
    print("test down!")
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
