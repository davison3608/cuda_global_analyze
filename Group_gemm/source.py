import torch

def group_gemm(configs: list, iter: int):
    results = []
    alpha = 1.0
    beta = 0.0
    for _ in range(iter):
        for i, (M, N, K) in enumerate(configs):
            print(f"Group {i}: M={M}, N={N}, K={K}")

            # 随机生成半精度矩阵
            A = torch.randn(M, K, dtype=torch.float16, device='cuda')
            B = torch.randn(K, N, dtype=torch.float16, device='cuda')

            # 执行 C = A @ B
            C = torch.matmul(A, B) * alpha  # or A @ B
            C += C * beta
            results.append(C)
            print(f"  Output shape: {C.shape}")

if __name__ == "__main__":
    configs = [
        (1024, 256, 512), # M N K
        (512, 128, 1024),
        (2048, 1024, 128),
        (256, 512, 256)
    ]
    group_gemm(configs, iter=1)

