#include <iostream>
#include <opencv2/opencv.hpp>
#include "CudaKernel.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) 
{
    IplImage* image;

    image = cvLoadImage("4555472_460s.jpg", CV_LOAD_IMAGE_GRAYSCALE);

    if(!image )
    {
        cout << "Could not open or find the image" << std::endl;
        return -1;
    }


    IplImage* image2 = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,image->nChannels);
    IplImage* image3 = cvCreateImage(cvGetSize(image),IPL_DEPTH_32F,image->nChannels);

    //Convert the input image to float
    cvConvert(image,image3);

    float *output = (float*)image2->imageData;
    float *input =  (float*)image3->imageData;

    kernelcall(input, output, image->width,image->height, image3->widthStep);

    //Normalize the output values from 0.0 to 1.0
    cvScale(image2,image2,1.0/255.0);

    cvShowImage("Original Image", image );
    cvShowImage("Sobeled Image", image2);
    cvWaitKey(0);
    return 0;
}