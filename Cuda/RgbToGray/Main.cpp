#include <iostream>
#include <cuda_runtime.h> //uchar4
#include <opencv2/core/core.hpp> //opencv functions
#include <opencv2/opencv.hpp> //opencv constants

#include "Function.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]){
    uchar4   *h_rgbaImage, *d_rgbaImage; 
    unsigned char *h_greyImage, *d_greyImage;

    string input_file = string(argv[1]);
    string output_file = string(argv[2]);

    Mat image, imageRGBA; 
    image = imread(input_file.c_str(), CV_LOAD_IMAGE_COLOR);
    cvtColor(image, imageRGBA, CV_BGR2RGBA);
    h_rgbaImage = (uchar4 *) imageRGBA.ptr<unsigned char>(0);

    size_t numPixels = image.rows * image.cols;

    cudaMalloc(&d_rgbaImage, sizeof(uchar4) * numPixels);
    cudaMalloc(&d_greyImage, sizeof(unsigned char) * numPixels);

    cudaMemcpy(d_rgbaImage, h_rgbaImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

    function(d_rgbaImage, d_greyImage, image.rows, image.cols);

    Mat imageGray;
    imageGray.create(image.rows, image.cols, CV_8UC1);
    h_greyImage = imageGray.ptr<unsigned char>(0);
    cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost);

    Mat output(image.rows, image.cols, CV_8UC1, (void*) h_greyImage);
    imwrite(output_file.c_str(), output);

    cudaFree(d_rgbaImage);
    cudaFree(d_greyImage);
}
