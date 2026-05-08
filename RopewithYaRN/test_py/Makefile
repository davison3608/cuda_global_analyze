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

# 默认目标
all:
	@echo "Usage: make gptj | make neox"

# GPT-J 目标
gptj: gptj_rope.so

gptj_rope.so: ../rope_withyarn_gptj_v1.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

# NeoX 目标
neox: neox_rope.so

neox_rope.so: ../rope_withyarn_neox_v1.cu
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $<

prof:
	nvprof python3 rope_class.py

# 清理中间文件（不删除 .so！）
clean:
	rm -f *.ii *.bc *.ptx *.s *.o *.hipi *.hipfb *.txt *.out *.csv *.json *.db

.PHONY: all gptj neox clean prof
