#include "cukernel.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <torch/csrc/api/include/torch/nn.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

static inline cudaError_t kernel_lanch_v2(
    cudaStream_t&str, cudaEvent_t&st, cudaEvent_t&ed,
    half* x, half* res, 
    int&dim, float&eps, half* weights,
    half* out_res
) 
{
    printf("CatRMSNorm v2\n");
    float* tmp_variance = nullptr;
    cudaMallocAsync(&tmp_variance, sizeof(float), str);
    cudaMemsetAsync(tmp_variance, 0.0f, sizeof(float), str);

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCKSIZE = BLOCK_Y * BLOCK_X;
    dim3 block(BLOCK_X, BLOCK_Y);
    int GRID_X = (dim + BLOCKSIZE - 1) / BLOCKSIZE;
    int GRID_Y = 1;
    dim3 grid(GRID_X, GRID_Y);

    cudaEventRecord(st, str);
    cudaStreamSynchronize(str);
    void* kernelArgs[] = {&x, &res, &dim, &eps, &weights, &out_res, &tmp_variance};
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v2<BLOCK_X, BLOCK_Y>,
        grid, block,
        kernelArgs, 
        0, str
    );  
    CUDA_CHECK(cudaStreamSynchronize(str));
    cudaEventRecord(ed, str);

    cudaFree(tmp_variance);
    return cudaGetLastError();
}

static inline cudaError_t kernel_lanch_v3(
    cudaStream_t&str, cudaEvent_t&st, cudaEvent_t&ed,
    half* x, half* res, 
    int&dim, float&eps, half* weights,
    half* out_res
) 
{
    printf("CatRMSNorm v3\n");
    float* tmp_variance = nullptr;
    cudaMallocAsync(&tmp_variance, sizeof(float), str);
    cudaMemsetAsync(tmp_variance, 0.0f, sizeof(float), str);

    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    constexpr int Vec_len = 4;
    constexpr int BLOCKSIZE = BLOCK_Y * BLOCK_X;
    dim3 block(BLOCK_X, BLOCK_Y);
    int GRID_X = (dim + BLOCKSIZE * Vec_len - 1) / (BLOCKSIZE * Vec_len);
    int GRID_Y = 1;
    dim3 grid(GRID_X, GRID_Y);

    cudaEventRecord(st, str);
    cudaStreamSynchronize(str);
    void* kernelArgs[] = {&x, &res, &dim, &eps, &weights, &out_res, &tmp_variance};
    cudaLaunchCooperativeKernel(
        (void*)groups_cat_rmsnorm_v3<BLOCK_X, BLOCK_Y, 4>,
        grid, block,
        kernelArgs, 
        0, str
    );  
    CUDA_CHECK(cudaStreamSynchronize(str));
    cudaEventRecord(ed, str);

    cudaFree(tmp_variance);
    return cudaGetLastError();
}

static inline cudaError_t kernel_lanch_v4(
    cudaStream_t&str, cudaEvent_t&st, cudaEvent_t&ed,
    half* x, half* res, 
    int&dim, float&eps, half* weights,
    half* out_res
) 
{
    printf("CatRMSNorm v4\n");
    constexpr int BLOCK_X = 32;
    constexpr int BLOCK_Y = 16;
    
    cudaEventRecord(st, str);
    groups_cat_rmsnorm_v4<BLOCK_X, BLOCK_Y>(
        x, res, 
        dim, eps,
        weights,
        out_res,
        str
    );
    cudaEventRecord(ed, str);
    return cudaGetLastError();
}

/**
 * \param x 输入的原始向量 [512, ] [1536, ] [7168, ] half
 * \param residul 输入残差向量 与x同形状 half
 * \param dim 目标特征维度
 * \param eps 数值稳定项
 * \param weights 缩放权重参数 与x同形状 half
 * 
 * \returns 返回归一化张量 残差连接张量(不同内存区域)
*/
std::tuple<torch::Tensor, torch::Tensor> catrmsnorm(
    torch::Tensor&x,
    torch::Tensor&residual,
    int32_t dim,
    float eps,
    torch::Tensor&weights
)
{
    //获取张量指针与设备信息
    at::Half* dev_x = x.data_ptr<at::Half>();
    at::Half* dev_res = residual.data_ptr<at::Half>();
    at::Half* dev_weights = weights.data_ptr<at::Half>();

    //检查张量
    assert(x.size(-1) == dim && "输入向量长度应与目标特征维度一致");
    assert(residual.size(-1) == dim && "残差向量长度应与目标特征维度一致");
    assert(x.is_cpu() != true && "输入向量应在gpu端");
    assert(residual.is_cpu() != true && "残差向量应在gpu端");
    assert(weights.is_cpu() != true && "缩放权重应在gpu端");

    cudaStream_t cu_str;
    cudaEvent_t cu_st, cu_ed;
    cudaStreamCreateWithFlags(&cu_str, cudaStreamNonBlocking);
    cudaEventCreate(&cu_st);
    cudaEventCreate(&cu_ed);
    CUDA_CHECK(cudaGetLastError());

    half* out_res = nullptr;
    cudaMallocAsync(&out_res, sizeof(half) * dim, cu_str);
    cudaMemsetAsync(out_res, 0, sizeof(half) * dim, cu_str);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(kernel_lanch_v4(
        cu_str, cu_st, cu_ed,
        reinterpret_cast<half*>(dev_x), reinterpret_cast<half*>(dev_res),
        dim, eps, reinterpret_cast<half*>(dev_weights),
        out_res
    ));

    cudaEventDestroy(cu_st);
    cudaEventDestroy(cu_ed);
    cudaStreamDestroy(cu_str);

    torch::TensorOptions opts = torch::TensorOptions()
        .device(x.device())    
        .dtype(torch::kFloat16)
        .requires_grad(false); 
    torch::Tensor res_tensor = torch::from_blob(
        out_res, //已分配内存的指针
        {dim}, //张量形状
        [](void* ptr) {
            if (ptr)
                cudaFree(ptr);
        },
        opts //设备类型
    );
    return std::make_tuple(x, res_tensor);
}

//导出动态链接
namespace py = pybind11;
PYBIND11_MODULE(cu_catrmsnorm, m) {  
    m.def(
    "catrmsnorm",                                          
    &catrmsnorm,                                           
    py::arg("x"),                                          
    py::arg("residual"),
    py::arg("dim"),
    py::arg("eps"),
    py::arg("weights"),
    "Perform CAT RMSNorm on CUDA tensors with residual connection" 
    );
}
