#include "stdio.h"
#include "stdlib.h"
#include "patch2col.h"



void patch2col_cpu(const float* data_img, const int channels, const int height, const int width, float* col_data)
{
    const int ks = 3;
	int out_width = width - ks + 1;
    int col_width = ks * ks * channels;
    int ch_page_size = width * height;
	for (int y = 0; y < height - ks + 1; y++) {
		for (int x = 0; x < width - ks + 1; x++) {
			float* out = col_data + (y * out_width + x) * col_width;
			int offset = 0;
			for (int chidx = 0; chidx < channels; chidx++) {
				const float* in = data_img + chidx * ch_page_size;
				for (int dy = 0; dy < ks; dy++) {
					for (int dx = 0; dx < ks; dx++) {
						out[offset] = in[(y + dy) * width + (x + dx)];
						offset++;
					}
				}
			}
		}
	}
    return;
}

void patch2col(const int dev, const float* data_img, const int channels, const int height, const int width, float* col_data)
{
    if(dev == 0)
    {
        patch2col_cpu(data_img, channels, height, width, col_data);
    }
    return;
}