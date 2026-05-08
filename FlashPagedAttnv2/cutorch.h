#pragma once
#include <iostream>
#include <memory>
#include <random>
#include <cmath>
#include <stdlib.h>
#include <stdio.h>
#include <array>
#include <tuple>
#include <atomic>
#include <thread>
#include <vector>
#include <condition_variable>
#include <pthread.h>
#include <numeric>
#include <future>
#include <optional>
#include <mutex>
#include <shared_mutex>
#include <assert.h>
#include <queue>
#include <deque>
#include <functional>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>         
#include <vector_types.h>         
#include <device_launch_parameters.h>  
#include <host_config.h>
#include <host_defines.h>

#include <cuda/atomic> 
#include <cuda/barrier>
#include <cuda/pipeline>

#include "cooperative_groups.h"
#include "cooperative_groups/memcpy_async.h"
#include "cooperative_groups/reduce.h"
#include "cooperative_groups/scan.h"

#define CUDA_CHECK(call) do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA 错误: %s 失败\n", #call); \
fprintf(stderr, "错误代码: %d\n", err); \
fprintf(stderr, "错误描述: %s\n", cudaGetErrorString(err)); \
fprintf(stderr, "文件: %s, 行号: %d\n", __FILE__, __LINE__); \
exit(EXIT_FAILURE); \
} \
} while (0)

#include <cublas.h>
#include <cudnn.h>
#include <mma.h>
#include <nccl.h>
//#include <openmpi/mpi.h>

#define CHECK_NCCL(err) do { \
if (err != ncclSuccess) { \
std::cerr << "NCCL error at " \
<< __FILE__ << ":" \
<< __LINE__ \
<< " error: " \
<< ncclGetErrorString(err) \
<< std::endl; \
exit(EXIT_FAILURE); \
} \
} while (0)

#define CHECK_CUDNN(err) \
if (err != CUDNN_STATUS_SUCCESS) { \
printf("cuDNN error: %s at line %d\n", cudnnGetErrorString(err), __LINE__); \
exit(EXIT_FAILURE); \
}
