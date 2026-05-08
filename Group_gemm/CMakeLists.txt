cmake_minimum_required(VERSION 3.20)
project(exec_test LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_INCLUDE_PATH "/usr/local/cuda/include")  
set(CUDA_LIB_PATH "/usr/local/cuda/lib64")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -w")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -gencode arch=compute_75,code=sm_75"
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O2")  
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -Xcompiler -w -Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_ARCHITECTURES 75)

set(SOURCES 
    test.cu
    group_gemm_merge_v1.cu
    group_gemm_merge_v2.cu
    group_gemm_merge_v3_withpair.cu
    group_gemm_merge_v3_vecload.cu
    group_gemm_merge_v4_vecloadcache.cu
    group_gemm_merge_v5_vecloadcache_core.cu
)

add_executable(exec_test ${SOURCES})
target_link_libraries(exec_test PRIVATE cudart cuda)

# 强制添加CUDA头文件路径
target_include_directories(exec_test PRIVATE ${CUDA_INCLUDE_PATH})

# 强制链接CUDA库路径
link_directories(${CUDA_LIB_PATH})
target_link_libraries(exec_test PRIVATE cudart)  # 链接CUDA运行时库