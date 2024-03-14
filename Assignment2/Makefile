NVCC := nvcc

CXXFLAGS := -I./lib

all: src/delta-step-cpu src/delta-step-gpu

src/delta-step-cpu: src/main.o src/cpu.o lib/mmio.o
	$(CXX) -o src/delta-step-cpu src/main.o src/cpu.o lib/mmio.o

src/delta-step-gpu: src/main.o src/gpu.o lib/mmio.o
	$(NVCC) -o src/delta-step-gpu src/main.o src/gpu.o lib/mmio.o

src/gpu.o: src/gpu.cu
	$(NVCC) -o src/gpu.o -c src/gpu.cu

clean:
	rm -f \
		lib/mmio.o \
		src/main.o \
		src/cpu.o \
		src/gpu.o \
		src/delta-step-cpu \
		src/delta-step-gpu