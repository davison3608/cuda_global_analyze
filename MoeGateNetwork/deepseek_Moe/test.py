import torch
import time
from dp_Moe import MoE
from dp_Moe import ModelArgs

def main():
    args = ModelArgs()
    test_moe = MoE(args=args)

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    dtype = torch.float16
    test_moe = test_moe.to(device="cuda", dtype=dtype)
    test_moe.eval()

    dim = args.dim
    batchsize = 1
    x = torch.randn(batchsize, 1, dim, dtype=dtype, device="cuda")

    totall_time = 0
    max_seqlen = 4096
    with torch.no_grad():
        for step in range(max_seqlen):
            start = time.time()
            # MoE前向传播
            moe_output = test_moe(x)  # 输出形状和输入一致：[batch_size, current_seq_len, dim]
            end = time.time()

            # 取最后一个 Token 的特征：[batch_size, 1, dim]
            # 实际大模型中这里会经过 Softmax 选 Token ID，再通过 Embedding 层转特征，这里直接取最后一个 Token 的特征作为下一个
            next_token_feat = moe_output[:, -1:, :]

            # 拼接序列 将新生成的Token特征拼接到输入中，模拟序列增长
            x = torch.cat([x, next_token_feat], dim=1)

            # 打印进度
            current_seq_len = x.shape[1]
            print(f"解码步数：{step} MoE 输出形状：{moe_output.shape}")
            print(f"当前计时: {(end - start) * 1000} MS")
            totall_time += end - start
    print(f"总计时: {totall_time}")
    print(f"平均计时: {(totall_time / max_seqlen) * 1000} MS")

if __name__=="__main__":
    main()
