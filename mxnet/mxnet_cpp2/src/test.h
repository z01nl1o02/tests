#pragma once

#ifdef calc_EXPORTS
#define CALC_API __declspec(dllexport) 
#else
#define CALC_API __declspec(dllimport) 
#endif
extern "C"
{
	CALC_API int c_calc_sum(float* matA, float* matB, float* matC, int width, int height);
};