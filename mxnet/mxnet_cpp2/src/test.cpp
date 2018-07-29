#include "test.h"

int c_calc_sum(float* matA, float* matB, float* matC, int width, int height)
{
    for( int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            matC[y * width + x] = matA[y * width + x] + matB[y * width + x];
        }
    }
    return 0;
}