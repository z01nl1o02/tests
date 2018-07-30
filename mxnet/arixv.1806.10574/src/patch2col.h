#pragma once

#include "def.h"

extern "C"
{
	CHELP_API void patch2col(const int dev, const float* data_img, const int channels, const int height, const int width, float* col_data);
};