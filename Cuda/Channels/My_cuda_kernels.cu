#include <cuda_runtime.h> //uchar4

__global__
void split_channels(uchar4 *input_image, uchar4 *red, uchar4 *green, uchar4 *blue){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    red[idx] = input_image[idx];
    green[idx] = input_image[idx];
    blue[idx] = input_image[idx];

    red[idx].x = 0;
    red[idx].y = 0;

    green[idx].x = 0;
    green[idx].z = 0;

    blue[idx].y = 0;
    blue[idx].z = 0;
}

void separate_channels(uchar4 *input_image, uchar4 *red, uchar4 *green, uchar4 *blue){
    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    split_channels<<<blockSize, threadSize>>>(input_image, red, green, blue);
}
