# Use nvcc for both CPU and GPU code
CXX := nvcc
NVCC := nvcc

CXXFLAGS := -Xcompiler -fopenmp -g
CUFLAGS := -G

all: inputgen/inputgen sorter/sorter

inputgen/inputgen:

sorter/sorter: sorter/sorter.o sorter/cpu/cpu.o sorter/gpu/gpu.o
	$(NVCC) -lgomp $^ -o $@

sorter/cpu/cpu.o: sorter/cpu/cpu.cpp sorter/cpu/cpu.h

sorter/gpu/gpu.o: sorter/gpu/gpu.cu sorter/gpu/gpu.h
	$(NVCC) $(CUFLAGS) -c $< -o $@

clean:
	rm -f \
		inputgen/inputgen \
		sorter/sorter \
		sorter/cpu/cpu.o \
		sorter/gpu/gpu.o

.PHONY: all clean