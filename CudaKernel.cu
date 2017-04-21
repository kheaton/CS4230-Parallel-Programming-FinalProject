#include<cuda.h>
#include<iostream>
#include "CudaKernel.h"

using namespace std;

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define checkCudaErrors(err) __checkCudaErrors (err, __FILE__, __LINE__)


texture <float,2,cudaReadModeElementType> tex1;

static cudaArray *cuArray = NULL;

//Kernel for x direction sobel
__global__ void implement_x_sobel(float* output,int width,int height,int widthStep)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //Make sure that thread is inside image bounds
    if(x<width && y<height)
    {
        float output_value = (-1*tex2D(tex1,x-1,y-1)) + (0*tex2D(tex1,x,y-1)) + (1*tex2D(tex1,x+1,y-1))
                           + (-2*tex2D(tex1,x-1,y))   + (0*tex2D(tex1,x,y))   + (2*tex2D(tex1,x+1,y))
                           + (-1*tex2D(tex1,x-1,y+1)) + (0*tex2D(tex1,x,y+1)) + (1*tex2D(tex1,x+1,y+1));

        output[y*widthStep+x]=output_value;
    }

}


inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
            file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}   

//Host Code
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        printf("cudaSafeCall() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }    
#endif

    return;
}
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        printf("cudaCheckError() failed at %s:%i : %s\n",
            file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

void kernelcall(float* input,float* output,int width,int height,int widthStep)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();

    CudaSafeCall(cudaMallocArray(&cuArray,&channelDesc,width,height));

    //Never use 1D memory copy if host and device pointers have different widthStep.
    // You don't know the width step of CUDA array, so its better to use cudaMemcpy2D...
    cudaMemcpy2DToArray(cuArray,0,0,input,widthStep,width * sizeof(float),height,cudaMemcpyHostToDevice);

    cudaBindTextureToArray(tex1,cuArray,channelDesc);

    float * D_output_x;
    CudaSafeCall(cudaMalloc(&D_output_x,widthStep*height)); 

    dim3 blocksize(16,16);
    dim3 gridsize;
    gridsize.x=(width+blocksize.x-1)/blocksize.x;
    gridsize.y=(height+blocksize.y-1)/blocksize.y;

    implement_x_sobel<<<gridsize,blocksize>>>(D_output_x,width,height,widthStep/sizeof(float));

    cudaThreadSynchronize();
    CudaCheckError();

    //Don't forget to unbind the texture
    cudaUnbindTexture(tex1);

    CudaSafeCall(cudaMemcpy(output,D_output_x,height*widthStep,cudaMemcpyDeviceToHost));

    cudaFree(D_output_x);
    cudaFreeArray(cuArray);
}