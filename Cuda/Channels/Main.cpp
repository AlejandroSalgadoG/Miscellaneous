#include <iostream>
#include <cuda_runtime.h> //uchar4
#include <opencv2/core/core.hpp> //opencv functions
#include <opencv2/opencv.hpp> //opencv constants

#include "My_cuda_kernels.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    uchar4 *h_inputImageRGBA, *d_inputImageRGBA;
    uchar4 *h_red, *h_green, *h_blue;
    uchar4 *d_red, *d_green, *d_blue;

    string input_file = string(argv[1]);

    Mat image, imageInputRGBA;
    image = imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    cvtColor(image, imageInputRGBA, CV_BGR2RGBA);
    h_inputImageRGBA = (uchar4 *) imageInputRGBA.ptr<unsigned char>(0);

    size_t numPixels = image.rows * image.cols;

    cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_red, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_green, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_blue, sizeof(uchar4) * numPixels);

    cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

    separate_channels(d_inputImageRGBA, d_red, d_green, d_blue);

    Mat red, green, blue;
    red.create(image.rows, image.cols, CV_8UC4);
    green.create(image.rows, image.cols, CV_8UC4);
    blue.create(image.rows, image.cols, CV_8UC4);

    h_red = (uchar4 *) red.ptr<unsigned char>(0);
    h_green = (uchar4 *) green.ptr<unsigned char>(0);
    h_blue = (uchar4 *) blue.ptr<unsigned char>(0);

    cudaMemcpy(h_red, d_red, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_green, d_green, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blue, d_blue, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

    Mat outputR(image.rows, image.cols, CV_8UC4, (void*) h_red);
    Mat outputG(image.rows, image.cols, CV_8UC4, (void*) h_green);
    Mat outputB(image.rows, image.cols, CV_8UC4, (void*) h_blue);

    imwrite("ImageR.png", outputR);
    imwrite("ImageG.png", outputG);
    imwrite("ImageB.png", outputB);

    cudaFree(d_inputImageRGBA);
    cudaFree(d_red);
    cudaFree(d_green);
    cudaFree(d_blue);
}
