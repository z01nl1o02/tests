#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include "opencv2\\opencv.hpp"
#include "stationary-wavelet-transform.h"
using namespace std;
using namespace cv;


void save_coef(vector<Mat>& vCoef)
{
	unsigned long k;
	ostringstream oss;
	for(k = 0; k < vCoef.size(); k++)
	{
		Mat m;
		Mat coef = vCoef[k];
		convertScaleAbs(coef, m);
		oss.str("");
		oss<<k<<".jpg";
		imwrite(oss.str(),m);
	}
}


int main(int argc, char* argv[])
{
	Mat img = imread(argv[1],0);
	vector<Mat> vCoef;
	stationary_wavelet_transform(img, vCoef);

	save_coef(vCoef);

	return 0;
}

