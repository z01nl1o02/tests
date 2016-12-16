#include "stationary-wavelet-transform.h"

void dwt2(Mat& gray,
	Mat& filterL, Mat& filterH,
	Mat& A, Mat& H, 
	Mat& V, Mat& D
	)
{
	int fs = filterL.cols;
	A = Mat_<float>(gray.rows/fs, gray.cols/fs);
	H = Mat_<float>(gray.rows/fs, gray.cols/fs);
	V = Mat_<float>(gray.rows/fs, gray.cols/fs);
	D = Mat_<float>(gray.rows/fs, gray.cols/fs);

	Mat tA = Mat_<float>(gray.rows, gray.cols/fs);
	Mat tH = Mat_<float>(gray.rows, gray.cols/fs);
	int x,y;
	for(y = 0; y < gray.rows; y++)
	{
		for(x = 0; x + fs < gray.cols; x += fs)
		{
			Mat m = gray( Rect(x,y,fs,1) );
			tA.at<float>(y,x/fs) = sum(m.mul(filterL)).val[0];
			tH.at<float>(y,x/fs) = sum(m.mul(filterH)).val[0];
		}
	}


	for(y = 0; y + fs < tA.rows; y += fs)
	{
		for(x = 0; x < tA.cols; x ++)
		{
			Mat m = tA( Rect(x,y,1,fs) ).t();
			A.at<float>(y / fs, x) = sum(m.mul(filterL)).val[0];
			V.at<float>(y / fs, x) = sum(m.mul(filterH)).val[0];



			m = tH( Rect(x,y,1,fs) ).t();
			H.at<float>(y / fs, x) = sum(m.mul(filterL)).val[0];
			D.at<float>(y / fs, x) = sum(m.mul(filterH)).val[0];

		}
	}
	return;
}

void haar_filter(Mat& filterL, Mat& filterH)
{
	float dataL[] = {0.5,0.5};
	float dataH[] = {-0.5,0.5};
	filterL = Mat_<float>(1,2);
	filterL.at<float>(0,0) = dataL[0];
	filterL.at<float>(0,1) = dataL[1];

	filterH = Mat_<float>(1,2);
	filterH.at<float>(0,0) = dataH[0];
	filterH.at<float>(0,1) = dataH[1];


	return;
}
void stationary_wavelet_transform(Mat& gray, vector<Mat>& vCoef)
{
	Mat floatGray;
	gray.convertTo(floatGray,CV_32F);
	Mat filterL, filterH;
	haar_filter(filterL, filterH);

	Mat A, H, V, D;
	do
	{
		if(filterL.cols * 32 >= floatGray.cols || filterL.rows * 32 >= floatGray.rows)
		{
			vCoef.push_back(A.clone());
			break;
		}
		dwt2(floatGray, filterL, filterH, A, H, V, D);
		vCoef.push_back(H.clone());
		vCoef.push_back(V.clone());
		vCoef.push_back(D.clone());
		int len = filterL.cols * 2;
		resize(filterL, filterL, Size(len,1));
		resize(filterH, filterH, Size(len,1));
	}while(1);
}