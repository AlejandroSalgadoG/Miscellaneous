//          Red             Green               Blue         Transparency
//uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w);

//I = .299f * R + .587f * G + .114f * B

__global__
void rgba_to_greyscale(uchar4 *rgbaImage, unsigned char *greyImage, int num_rows, int num_cols){
    int row = threadIdx.x;
    int col = blockIdx.x;
    int idx = col + row*360;

    greyImage[idx] = 0.299f * rgbaImage[idx].x + 0.587f * rgbaImage[idx].y + 0.114f * rgbaImage[idx].z; 
}

void function(uchar4 *d_rgbaImage, unsigned char *d_greyImage, size_t num_rows, size_t num_cols){
    dim3 blockSize(360,1,1);
    dim3 threadSize(480,1,1);

    rgba_to_greyscale<<<blockSize, threadSize>>>(d_rgbaImage, d_greyImage, num_rows, num_cols);
}
