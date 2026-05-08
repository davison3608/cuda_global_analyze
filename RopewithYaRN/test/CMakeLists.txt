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
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w -Xcompiler -w -Wno-deprecated-gpu-targets")

set(CMAKE_CUDA_ARCHITECTURES 75)

set(SOURCES 
    test.cu
    rope_withyarn_gptj_v1.cu
    rope_withyarn_neox_v1.cu
    rope_withyarn_gptj_v1_tokens.cu
    rope_withyarn_neox_v1_tokens.cu
)

# --- 路径配置 ---
set(PYTHON_INCLUDE_DIR "/home/davison/anaconda3/envs/stable_diffusion/include/python3.10")
set(PYTORCH_PREFIX "/home/davison/anaconda3/envs/stable_diffusion/lib/python3.10/site-packages/torch")
set(PYBIND11_INCLUDE_DIR "${PYTORCH_PREFIX}/include")
set(TORCH_INCLUDE_DIRS 
    "${PYTORCH_PREFIX}/include"
    "${PYTORCH_PREFIX}/include/torch/csrc/api/include"
)

# --- Python 库路径（关键！）---
set(PYTHON_LIBRARY "/home/davison/anaconda3/envs/stable_diffusion/lib/libpython3.10.so")

add_executable(exec_test ${SOURCES})

target_include_directories(exec_test PRIVATE
    ${CUDA_INCLUDE_PATH}
    ${PYBIND11_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIR}   
)

# --- 链接目录与库 ---
link_directories(${CUDA_LIB_PATH})

target_link_libraries(exec_test PRIVATE
    cudart
    cuda
    ${PYTHON_LIBRARY}   # ← 必须链接 Python 解释器
    dl                  # ← 动态加载支持
    pthread             # ← 线程支持（GIL 需要）
    util                # ← 辅助函数（如 forkpty）
)

target_compile_definitions(exec_test PRIVATE
    PYBIND11_USE_CUDA
)
