INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=sobel.cu
EXECUTABLE=sobel
EXTENSION=.out

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@$(EXTENSION) -Wno-deprecated-gpu-targets

clean:
	rm -f *.out

ready:
	module load cuda

run:
	sbatch sobel.gpu

check:
	squeue -u u0517990