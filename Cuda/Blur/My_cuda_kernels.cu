#include <cuda_runtime.h> //uchar4

__global__
void split_channels(uchar4 *input_image, unsigned char *red, unsigned char *green, unsigned char *blue){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    red[idx] = input_image[idx].x;
    green[idx] = input_image[idx].y;
    blue[idx] = input_image[idx].z;
}

__global__
void combine_channels(uchar4 *output_image, unsigned char *red, unsigned char *green, unsigned char *blue){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    output_image[idx] = make_uchar4(red[idx], green[idx], blue[idx], 255);
}

__global__
void blur_channel(unsigned char *channel, float *filter, int filter_size){

} 

void separate_channels(uchar4 *input_image, unsigned char *red, unsigned char *green, unsigned char *blue){
    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    split_channels<<<blockSize, threadSize>>>(input_image, red, green, blue);
}

void recombine_channels(uchar4 *output_image, unsigned char *red, unsigned char *green, unsigned char *blue){
    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    combine_channels<<<blockSize, threadSize>>>(output_image, red, green, blue);
}

void blur_image(unsigned char *red, unsigned char *green, unsigned char *blue, float *filter, int filter_size){
    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    blur_channel<<<blockSize, threadSize>>>(red, filter, filter_size);
    blur_channel<<<blockSize, threadSize>>>(green, filter, filter_size);
    blur_channel<<<blockSize, threadSize>>>(blue, filter, filter_size);
}
