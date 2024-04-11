# Use nvcc for both CPU and GPU code
CXX := nvcc
NVCC := nvcc

CXXFLAGS := -Xcompiler -fopenmp
CUFLAGS := 

all: inputgen/inputgen sorter/sorter verifier/verifier

inputgen/inputgen:

sorter/sorter: sorter/sorter.o sorter/cpu/cpu.o sorter/gpu/gpu.o
	$(NVCC) -lgomp $^ -o $@

sorter/cpu/cpu.o: sorter/cpu/cpu.cpp sorter/cpu/cpu.h

sorter/gpu/gpu.o: sorter/gpu/gpu.cu sorter/gpu/gpu.h
	$(NVCC) $(CUFLAGS) -c $< -o $@

verifier/verifier:

clean:
	rm -f \
		inputgen/inputgen \
		sorter/sorter \
		verifier/verifier \
		sorter/sorter.o \
		sorter/cpu/cpu.o \
		sorter/gpu/gpu.o

.PHONY: all clean