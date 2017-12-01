#pragma once

void separate_channels(uchar4 *input_image, unsigned char *red, unsigned char *green, unsigned char *blue);
void recombine_channels(uchar4 *output_image, unsigned char *red, unsigned char *green, unsigned char *blue);

void blur_image(unsigned char *red, unsigned char *green, unsigned char *blue, float *filter, int filter_size);
