// Kyle Heaton
// U0517990
// Final Project

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include "string.h"

#define DEFAULT_THRESHOLD  4000
#define DEFAULT_FILENAME "testing-image.ppm"

unsigned int *read_ppm( char *filename, int * xsize, int * ysize, int *maxval ){
  
	if ( !filename || filename[0] == '\0') {
		fprintf(stderr, "read_ppm but no file name\n");
		return NULL;  // fail
	}

  	FILE *fp;

	fprintf(stderr, "read_ppm( %s )\n", filename);
	fp = fopen( filename, "rb");
	if (!fp) 
	{
		fprintf(stderr, "read_ppm()    ERROR  file '%s' cannot be opened for reading\n", filename);
		return NULL; // fail 
	}

	char chars[1024];
	//int num = read(fd, chars, 1000);
	int num = fread(chars, sizeof(char), 1000, fp);

	if (chars[0] != 'P' || chars[1] != '6') 
	{
		fprintf(stderr, "Texture::Texture()    ERROR  file '%s' does not start with \"P6\"  I am expecting a binary PPM file\n", filename);
		return NULL;
	}

	unsigned int width, height, maxvalue;


	char *ptr = chars+3; // P 6 newline
	if (*ptr == '#') // comment line! 
	{
		ptr = 1 + strstr(ptr, "\n");
	}

	num = sscanf(ptr, "%d\n%d\n%d",  &width, &height, &maxvalue);
	fprintf(stderr, "read %d things   width %d  height %d  maxval %d\n", num, width, height, maxvalue);  
	*xsize = width;
	*ysize = height;
	*maxval = maxvalue;
  
	unsigned int *pic = (unsigned int *)malloc( width * height * sizeof(unsigned int));
	if (!pic) {
		fprintf(stderr, "read_ppm()  unable to allocate %d x %d unsigned ints for the picture\n", width, height);
		return NULL; // fail but return
	}

	// allocate buffer to read the rest of the file into
	int bufsize =  3 * width * height * sizeof(unsigned char);
	if ((*maxval) > 255) bufsize *= 2;
	unsigned char *buf = (unsigned char *)malloc( bufsize );
	if (!buf) {
		fprintf(stderr, "read_ppm()  unable to allocate %d bytes of read buffer\n", bufsize);
		return NULL; // fail but return
	}


	// TODO really read
	char duh[80];
	char *line = chars;

	// find the start of the pixel data.   no doubt stupid
	sprintf(duh, "%d\0", *xsize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *ysize);
	line = strstr(line, duh);
	//fprintf(stderr, "%s found at offset %d\n", duh, line-chars);
	line += strlen(duh) + 1;

	sprintf(duh, "%d\0", *maxval);
	line = strstr(line, duh);


	fprintf(stderr, "%s found at offset %d\n", duh, line - chars);
	line += strlen(duh) + 1;

	long offset = line - chars;
	//lseek(fd, offset, SEEK_SET); // move to the correct offset
	fseek(fp, offset, SEEK_SET); // move to the correct offset
	//long numread = read(fd, buf, bufsize);
	long numread = fread(buf, sizeof(char), bufsize, fp);
	fprintf(stderr, "Texture %s   read %ld of %ld bytes\n", filename, numread, bufsize); 

	fclose(fp);

	int pixels = (*xsize) * (*ysize);
	for (int i=0; i<pixels; i++) pic[i] = (int) buf[3*i];  // red channel

	return pic; // success
}

void write_ppm( char *filename, int xsize, int ysize, int maxval, int *pic) {
	FILE *fp;
	//int x,y;

	fprintf(stderr, "write_ppm( %s )\n", filename);	

	fp = fopen(filename, "w");
	if (!fp) 
	{
		fprintf(stderr, "FAILED TO OPEN FILE '%s' for writing\n", filename);
		exit(-1); 
	}

	fprintf(fp, "P6\n"); 
	fprintf(fp,"%d %d\n%d\n", xsize, ysize, maxval);

	int numpix = xsize * ysize;
	for (int i=0; i<numpix; i++) {
		unsigned char uc = (unsigned char) pic[i];
		fprintf(fp, "%c%c%c", uc, uc, uc); 
	}

	fclose(fp);
}

__global__ void sobel(int* imageWidth, int* imageHeight, int* image, int* output, int* threshold) {
	int width = *imageWidth;
	int height = *imageHeight;

	for (int i = 1;  i < height - 1; i++) {
		for (int j = 1; j < width -1; j++) {
			int offset = i * width + j;
			

			int sum1 =  image[width * (i - 1) + j + 1 ] - image[width * (i - 1) + j - 1] +
						2 * image[width * (i) + j + 1 ] - 2 * image[width * (i) + j - 1] +
						image[width * (i + 1) + j + 1 ] - image[width * (i + 1) + j - 1];

			int sum2 = image[width * (i - 1) + j - 1] + 2 * image[width * (i - 1) + j] +
					   image[width * (i - 1) + j + 1] - image[width * (i + 1) + j - 1] - 
				   	   2 * image[width * (i + 1) + j] - image[width * (i + 1) + j + 1];

			int magnitude =  sum1 * sum1 + sum2 * sum2;

			if (magnitude > *threshold) {
				output[offset] = 255;
			}
			else { 
				output[offset] = 0;
			}
		}
	}
}

int main( int argc, char **argv ) {
	int thresh = DEFAULT_THRESHOLD;
	int number_of_files = 20000;
	cudaEvent_t start_event, stop_event;
	float elapsed_time_gpu;

	if(argc > 1) {
		number_of_files = atoi(argv[1]);
	}

	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	cudaEventRecord(start_event, 0);  
	for(int i = 1; i <= number_of_files; i++) {
		char *in_filename = (char*)malloc(36 * sizeof(char));
		char *out_filename = (char*)malloc(36 * sizeof(char));

		sprintf(in_filename, "./sintel/sintel%03d.ppm", i);
		sprintf(out_filename, "./sintel-sobel/sintel-sobel%03d.ppm", i);

		int xsize, ysize, maxval;
		unsigned int *pic = read_ppm( in_filename, &xsize, &ysize, &maxval ); 

		int numbytes =  xsize * ysize * 3 * sizeof( int );
		int *result = (int *) malloc( numbytes );

		if (!result) { 
			fprintf(stderr, "sobel() unable to malloc %d bytes\n", numbytes);
			exit(-1); // fail
		}

		int *out = result;

		// Set initial values of result
		for (int col=0; col<ysize; col++) {
			for (int row=0; row<xsize; row++) { 
				*out++ = -1; 			
			}
		}

		int *imageWidth, *imageHeight, *image, *output, *threshold;

		cudaMalloc((void **)&imageWidth, sizeof(int));
		cudaMalloc((void **)&imageHeight, sizeof(int));
		cudaMalloc((void **)&image, numbytes);
		cudaMalloc((void **)&output, numbytes);
		cudaMalloc((void **)&threshold, sizeof(int));

		cudaMemcpy(imageWidth, &xsize, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(imageHeight, &ysize, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(image, pic, numbytes, cudaMemcpyHostToDevice);
		cudaMemcpy(output, result, numbytes, cudaMemcpyHostToDevice);
		cudaMemcpy(threshold, &thresh, sizeof(int), cudaMemcpyHostToDevice);
		
		sobel<<<1,1>>>(imageWidth, imageHeight, image, output, threshold);

		cudaMemcpy(result, output, numbytes, cudaMemcpyDeviceToHost);

		cudaFree(imageWidth);
		cudaFree(imageHeight);
		cudaFree(image);
		cudaFree(output);
		cudaFree(threshold);
			
		write_ppm( out_filename, xsize, ysize, 255, result);

		free(pic);
		free(result);
		free(out_filename);
		free(in_filename);
	}
	cudaEventRecord(stop_event, 0);

	cudaEventSynchronize(stop_event);
	cudaEventElapsedTime(&elapsed_time_gpu,start_event, stop_event);

	printf("Parallel Time: %.2f msec\n", elapsed_time_gpu);

	fprintf(stderr, "sobel done\n"); 
}

int main1(int argc, char** argv) {

	// *Optional* - call ffmpeg to split up video
	// pull image files
	// setup cuda parameters (splitting up blocks warps ect)
	
	// https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
	// Async push images to GPU, then process, then pull back processed images to host

	// save processed images

	// *Optional* - call ffmpeg to stitch images back into video

	return 0;
}