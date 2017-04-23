INCLUDE=-I/usr/local/cuda/8.0/cuda/include 

SOURCE=sobel.cu
EXECUTABLE=sobel
EXTENSION=.out

$(EXECUTABLE): $(SOURCE)
	nvcc $(INCLUDE) $< -o $@$(EXTENSION) -Wno-deprecated-gpu-targets

sobel-seq:
	nvcc $(INCLUDE) sobel-seq.cu -o sobel-seq$(EXTENSION) -Wno-deprecated-gpu-targets

build: sobel sobel-seq

clean:
	rm -f *.out

ready:
	module load cuda

run:
	mkdir -p sintel-sobel
	rm -r -f ./sintel-sobel/*
	mkdir -p sintel-sobel-seq
	rm -r -f ./sintel-sobel-seq/*
	sbatch sobel.slurm
	sbatch sobel-seq.slurm

check:
	squeue -u u0517990

download:
	wget http://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.720p.mkv

download-1080:
	wget http://ftp.nluug.nl/pub/graphics/blender/demo/movies/Sintel.2010.1080p.mkv

images-ppm:
	mkdir -p sintel
	rm -r -f ./sintel/*
	ffmpeg -i Sintel.2010.720p.mkv -vf fps=24 ./sintel/sintel%03d.ppm

run-images:
	sbatch images.slurm

movie:
	ffmpeg -f image2 -r 24 -i ./sintel/sintel%03d.ppm -pix_fmt yuv420p ./sintel.mp4

run-movie:
	sbatch sobel-movie.slurm

sobel-movie:
	ffmpeg -f image2 -r 24 -i ./sintel-sobel/sintel-sobel%03d.ppm -pix_fmt yuv420p ./sintel-sobel.mp4