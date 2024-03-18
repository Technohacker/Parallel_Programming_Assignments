NVCC := nvcc

CXXFLAGS := -I./lib

all: src/delta-step-cpu src/delta-step-gpu

lib/libutil.a: lib/common.o lib/mmio.o
	ar rcs lib/libutil.a lib/common.o lib/mmio.o

src/delta-step-cpu: src/main.o src/cpu.o lib/libutil.a
	$(CXX) -o src/delta-step-cpu src/main.o src/cpu.o lib/libutil.a

src/delta-step-gpu: src/main.o src/gpu.o lib/libutil.a
	$(NVCC) -o src/delta-step-gpu src/main.o src/gpu.o lib/libutil.a

src/gpu.o: src/gpu.cu
	$(NVCC) $(CXXFLAGS) -o src/gpu.o -c src/gpu.cu

clean:
	rm -f \
		lib/common.o \
		lib/mmio.o \
		lib/libutil.a \
		src/main.o \
		src/cpu.o \
		src/gpu.o \
		src/delta-step-cpu \
		src/delta-step-gpu