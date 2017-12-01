#include <iostream>
#include <cuda_runtime.h> //uchar4
#include <opencv2/core/core.hpp> //opencv functions
#include <opencv2/opencv.hpp> //opencv constants

#include "My_cuda_kernels.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
    uchar4 *h_outputImageRGBA, *d_outputImageRGBA;
    unsigned char *d_red, *d_green, *d_blue;
    unsigned char *d_red_blurred, *d_green_blurred, *d_blue_blurred;

    string input_file = string(argv[1]);
    string output_file = string(argv[2]);

    Mat image, imageInputRGBA, imageOutputRGBA;

    image = imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    cvtColor(image, imageInputRGBA, CV_BGR2RGBA);
    imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

    h_inputImageRGBA = (uchar4 *) imageInputRGBA.ptr<unsigned char>(0);
    h_outputImageRGBA = (uchar4 *) imageOutputRGBA.ptr<unsigned char>(0);

    size_t numPixels = image.rows * image.cols;

    cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numPixels);

    cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

    float *d_filter;
    float h_filter[9] = { 0.f,  0.2f, 0.f,
                          0.2f, 0.2f, 0.2f,
                          0.f,  0.2f, 0.f };
    int filter_size = 3;

    cudaMalloc(&d_filter, sizeof(float) * 9);

    cudaMemcpy(d_filter, h_filter, sizeof(float) * 9, cudaMemcpyHostToDevice);

    cudaMalloc(&d_red, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_green, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_blue, sizeof(uchar4) * numPixels);

    cudaMalloc(&d_red_blurred, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_green_blurred, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_blue_blurred, sizeof(uchar4) * numPixels);

    blur_image(d_inputImageRGBA, d_outputImageRGBA, d_red, d_red_blurred, d_green, d_green_blurred, d_blue, d_blue_blurred, d_filter, filter_size);

    cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

    Mat output(image.rows, image.cols, CV_8UC4, (void*) h_outputImageRGBA);
    cvtColor(output, imageOutputRGBA, CV_RGBA2BGRA);
    imwrite(output_file.c_str(), imageOutputRGBA);

    cudaFree(d_inputImageRGBA);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
}
