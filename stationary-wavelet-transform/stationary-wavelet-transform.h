#ifndef STATIONARY_WAVELET_TRANSFORM_H
#define STATIONARY_WAVELET_TRANSFORM_H
/*
https://en.wikipedia.org/wiki/Stationary_wavelet_transform
*/
#include <iostream>
#include <string>
#include <vector>
#include "opencv2\\opencv.hpp"
using namespace std;
using namespace cv;

void stationary_wavelet_transform(Mat& gray, vector<Mat>& vCoef);



#endif
