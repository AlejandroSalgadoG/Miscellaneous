#pragma once

void blur_image(uchar4 *input_image, uchar4 *output_image,
                unsigned char *red_in, unsigned char *red_out,
                unsigned char *green_in, unsigned char * green_out,
                unsigned char *blue_in, unsigned char *blue_out,
                float *filter, int filter_size);
