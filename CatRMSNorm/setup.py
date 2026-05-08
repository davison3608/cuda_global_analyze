import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(current_dir, "lib")
os.makedirs(lib_dir, exist_ok=True)

# 收集所有需要编译的 CUDA 和 C++ 源文件
sources = [
    "forward.cu",
    "groups_cat_rmsnorm_v1.cu",
    "groups_cat_rmsnorm_v2.cu",
    "groups_cat_rmsnorm_v3.cu",
    "groups_cat_rmsnorm_v4.cu",
]
# 确保每个文件路径正确
sources = [os.path.join(current_dir, src) for src in sources]

if __name__ == "__main__":
    setup(
        name='',
        version='1.0.0',
        description='CatRMSNorm CUDA Decoding Library',
        ext_modules=[
            CUDAExtension(
                name='cu_Catrmsnorm_complie_v4',  # 生成的模块名
                sources=sources,
                include_dirs=[current_dir],  # 包含当前目录的头文件
                extra_compile_args={
                    'cxx': [
                        '-O2',
                        '-march=native',
                        '-std=c++17',
                        '-w',  # 关闭 C++ 警告
                    ],
                    'nvcc': [
                        '-O2',
                        '-lineinfo',
                        '--expt-relaxed-constexpr',
                        '-use_fast_math',
                        '-std=c++17',
                        '-gencode=arch=compute_75,code=sm_75',
                        #'-gencode=arch=compute_80,code=sm_80',
                        #'-gencode=arch=compute_86,code=sm_86',
                        '-Xcompiler', '-march=native',
                        '-w',
                    ]
                }
            )
        ],
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        },
        zip_safe=False,
        script_args=['build_ext', f'--build-lib={lib_dir}', '-j8']  # 强制输出到 lib 目录
    )

    print(f"✅ 构建完成！.so 文件应该在: {lib_dir}/")

    # 检查生成的文件
    so_files = [f for f in os.listdir(lib_dir) if f.endswith('.so')]
    for so_file in so_files:
        print(f"找到: {so_file}")