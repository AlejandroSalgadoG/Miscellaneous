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
void blur_channel(unsigned char *channel_in, unsigned char *channel_out, float *filter, int filter_size){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    int sum = 0;
    int cnt = 0;

    if (row >= 0){ sum += channel_in[col + (row-1)*360] * filter[1]; cnt++; }
    if (col >= 0){ sum += channel_in[idx-1] * filter[3]; cnt++; }
    if (col <= 360){ sum += channel_in[idx+1] * filter[5]; cnt++; }
    if (row <= 480){ sum += channel_in[col + (row+1)*360] * filter[7]; cnt++; }

    channel_out[idx] = sum / cnt;
}

__global__
void combine_channels(uchar4 *output_image, unsigned char *red, unsigned char *green, unsigned char *blue){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    output_image[idx] = make_uchar4(red[idx], green[idx], blue[idx], 255);
}

void blur_image(uchar4 *input_image, uchar4 *output_image,
                unsigned char *red_in, unsigned char *red_out,
                unsigned char *green_in, unsigned char * green_out,
                unsigned char *blue_in, unsigned char *blue_out,
                float * filter, int filter_size){

    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    split_channels<<<blockSize, threadSize>>>(input_image, red_in, green_in, blue_in);

    blur_channel<<<blockSize, threadSize>>>(red_in, red_out, filter, filter_size);
    blur_channel<<<blockSize, threadSize>>>(green_in, green_out, filter, filter_size);
    blur_channel<<<blockSize, threadSize>>>(blue_in, blue_out, filter, filter_size);

    combine_channels<<<blockSize, threadSize>>>(output_image, red_out, green_out, blue_out);
}
