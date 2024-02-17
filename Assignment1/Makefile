CXX := mpic++
CXXFLAGS := -I./util

all: \
	bin/find_tolerance \
	bin/inputgen \
	bin/serial \
	bin/parallel_cannon \
	bin/parallel_summa

clean:
	rm \
		bin/find_tolerance \
		bin/inputgen \
		bin/serial \
		bin/parallel_cannon \
		bin/parallel_summa \
		util/matrix.o \
		util/matrix_io_serial.o \
		util/matrix_io_parallel.o \
		util/timing_serial.o \
		util/timing_parallel.o \
	|| true

bin/inputgen: util/matrix.o util/matrix_io_serial.o
bin/serial: util/matrix.o util/matrix_io_serial.o util/timing_serial.o
bin/parallel_cannon: util/matrix.o util/matrix_io_parallel.o util/timing_parallel.o
bin/parallel_summa: util/matrix.o util/matrix_io_parallel.o util/timing_parallel.o

generate_inputs: bin/inputgen
	mkdir -p results
	bin/inputgen 6400 1 results/in_6400
	bin/inputgen 12800 1 results/in_12800
	bin/inputgen 25600 1 results/in_25600

.PHONY: all clean generate_inputs
