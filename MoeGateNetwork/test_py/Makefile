# 编译器
NVCC = nvcc

# 公共编译选项
CXXFLAGS = -O2 \
           -g \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -Xcompiler -fPIC \
           -DCUDA_HAS_FP16=1 \
           -DCOMPILE_FOR_STANDALONE_DEBUG_TEST \
           -w

# 链接选项
LDFLAGS = -shared

# 头文件路径
INCLUDES = -I..

# 源文件（位于上级目录）
SRCS = ../GroupedTopkfused_fp16_v1.cu \
       ../GroupedTopkfused_fp32_v1.cu \
       ../launch_v1.cu

# 目标 .so 路径
TARGET = GroupedTopkfused.so

all: $(TARGET)

# 构建规则：不再依赖 lib
$(TARGET): $(SRCS)
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@

# 清理中间文件（不删除 .so！）
clean:
	rm -f *.ii *.bc *.ptx *.s *.o *.hipi *.hipfb *.txt *.out *.csv *.json *.db
