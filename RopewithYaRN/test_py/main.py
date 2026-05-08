import importlib
import importlib.util
import os
import torch
from RopewithYaRN.test_py.rope_class import DeepseekScalingRotaryEmbedding

max_position = 128_000
seq_len = 1
rotary_dim = 64

num_head_mla = 32
nope_dim_mla = 128  # 128
rope_dim_mla = 64  # 64
kv_lora_rank = 512  # 512

num_head_indexer = 64
head_dim_indexer = 128  # 128
nope_dim_indexer = head_dim_indexer - rotary_dim  # 64
rope_dim_indexer = rotary_dim  # 64

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

common_kwargs = {
    "head_size": rotary_dim,
    "rotary_dim": rotary_dim,
    "max_position_embeddings": max_position,
    "base": 10000.0,
    "scaling_factor": 1.75,
    "dtype": dtype,
    "extrapolation_factor": 2.125,
    "attn_factor": 1.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale": 1.125,
    "mscale_all_dim": 0.0,
    "reference": False,
}

# GPT-J 风格
rope_gptj = DeepseekScalingRotaryEmbedding(is_neox_style=False, **common_kwargs)
# NeoX 风格
rope_neox = DeepseekScalingRotaryEmbedding(is_neox_style=True, **common_kwargs)

def single_test(start_pos: int):
    print(f"iter={n / seq_len} start_pos={n}")
    arange_pos = torch.arange(start=start_pos, end=start_pos + seq_len, dtype=torch.int32, device=device)
    positions = arange_pos.broadcast_to([1, seq_len])
    assert positions.is_contiguous()

    # gptj neox randn

    # q.shape=torch.Size([1, 32, 192]), q_pe.shape=torch.Size([1, 32, 64])
    query = torch.randn([1, seq_len, num_head_mla, nope_dim_mla + rope_dim_mla], dtype=dtype, device=device)
    query_c = query.clone()
    # k.shape=torch.Size([1, 1, 576]), k_pe.shape=torch.Size([1, 1, 64])
    key = torch.randn([1, seq_len, 1, kv_lora_rank + rope_dim_mla], dtype=dtype, device=device)
    key_c = key.clone()
    # q_pe.stride()=(6144, 192, 1), q_pe.stride(0)=6144, q_pe.stride(1)=192, q_pe.stride(2)=1
    _, q_pe = torch.split(query, [nope_dim_mla, rope_dim_mla], dim=-1)
    _, q_pe_c = torch.split(query_c, [nope_dim_mla, rope_dim_mla], dim=-1)
    # k_pe.stride()=(576, 576, 1), k_pe.stride(0)=576, k_pe.stride(1)=576, k_pe.stride(2)=1
    _, k_pe = torch.split(key, [kv_lora_rank, rope_dim_mla], dim=-1)
    _, k_pe_c = torch.split(key_c, [kv_lora_rank, rope_dim_mla], dim=-1)
    # gptj
    assert q_pe.is_contiguous() is False and q_pe_c.is_contiguous() is False
    q_res, k_res = rope_gptj.forward_torch(
        positions=positions,
        query=q_pe, key=k_pe,
    )
    q_res_ref, k_res_ref = rope_gptj.forward_hip(
        positions=positions,
        query=q_pe_c, key=k_pe_c,
    )

    # assert
    atol = 1e-2
    if torch.allclose(k_res, k_res_ref, atol=atol) is False:
        print(k_res)
        print(k_res_ref)
        max_error = torch.abs(k_res - k_res_ref).max().item()
        print(f"max error {max_error}")
        raise ValueError
    if torch.allclose(q_res, q_res_ref, atol=atol) is False:
        print(q_res)
        print(q_res_ref)
        max_error = torch.abs(q_res - q_res_ref).max().item()
        print(f"max error {max_error}")
        raise ValueError

    # q.shape=torch.Size([1, 64, 128]), q_pe.shape=torch.Size([1, 64, 64])
    query_x = torch.randn([1, seq_len, num_head_indexer, head_dim_indexer], dtype=dtype, device=device)
    query_x_c = query_x.clone()
    # k.shape=torch.Size([1, 1, 128]), k_pe.shape=torch.Size([1, 1, 64])
    key_x = torch.randn([1, seq_len, 1, head_dim_indexer], dtype=dtype, device=device)
    key_x_c = key_x.clone()
    # q_pe.stride()=(8192, 128, 1), q_pe.stride(0)=8192, q_pe.stride(1)=128, q_pe.stride(2)=1
    _, q_pe_x = torch.split(query_x, [nope_dim_indexer, rope_dim_indexer], dim=-1)
    _, q_pe_x_c = torch.split(query_x_c, [nope_dim_indexer, rope_dim_indexer], dim=-1)
    # k_pe.stride()=(128, 128, 1), k_pe.stride(0)=128, k_pe.stride(1)=128, k_pe.stride(2)=1
    _, k_pe_x = torch.split(key_x, [nope_dim_indexer, rope_dim_indexer], dim=-1)
    _, k_pe_x_c = torch.split(key_x_c, [nope_dim_indexer, rope_dim_indexer], dim=-1)
    # neox
    assert q_pe_x.is_contiguous() is False and q_pe_x_c.is_contiguous() is False
    q_res_nex, k_res_nex = rope_neox.forward_torch(
        positions=positions,
        query=q_pe_x, key=k_pe_x,
    )
    q_res_ref_n, k_res_ref_n = rope_neox.forward_hip(
        positions=positions,
        query=q_pe_x_c, key=k_pe_x_c,
    )

    atol = 1e-2
    if torch.allclose(k_res_nex, k_res_ref_n, atol=atol) is False:
        print(k_res_nex)
        print(k_res_ref_n)
        max_error = torch.abs(k_res_nex - k_res_ref_n).max().item()
        print(f"max error {max_error}")
        raise ValueError
    if torch.allclose(q_res_nex, q_res_ref_n, atol=atol) is False:
        print(q_res_nex, q_res_ref_n)
        max_error = torch.abs(q_res_nex - q_res_ref_n).max().item()
        print(f"max error {max_error}")
        raise ValueError

    print(f"gptj neox qk pass")


if __name__ == "__main__":
    for n in range(0, seq_len * 5_000, seq_len):
        single_test(start_pos=n)
    torch.cuda.empty_cache()
    print("test down")
