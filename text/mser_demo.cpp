#include <iostream>
#include <vector>
#include <iostream>
#include <string>
#include <ostream>
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "opencv/cv.h"

using namespace std;
using namespace cv;


static const Vec3b bcolors[] =
{
    Vec3b(0,0,255),
    Vec3b(0,128,255),
    Vec3b(0,255,255),
    Vec3b(0,255,0),
    Vec3b(255,128,0),
    Vec3b(255,255,0),
    Vec3b(255,0,0),
    Vec3b(255,0,255),
    Vec3b(255,255,255)
};

int run(string imagepath)
{
	Mat imgRGB = imread(imagepath,1);
	if(imgRGB.empty())
	{
		cout<<"empty image"<<endl;
		return 0;
	}
	Mat imgHSV;
	cvtColor(imgRGB, imgHSV, COLOR_RGB2HSV);
	Mat imgH = Mat_<uchar>(imgRGB.rows, imgRGB.cols);
	Mat imgS = Mat_<uchar>(imgRGB.rows, imgRGB.cols);
	Mat imgV = Mat_<uchar>(imgRGB.rows, imgRGB.cols);
	int x,y;
	for(y = 0; y < imgRGB.rows; y++)
	{
		for(x = 0; x < imgRGB.cols; x++)
		{
			uchar h = imgHSV.at<uchar>(y,x*3);
			uchar s = imgHSV.at<uchar>(y,x*3 + 1);
			uchar v = imgHSV.at<uchar>(y,x*3 + 2);
			imgH.at<uchar>(y,x) = h;
			imgS.at<uchar>(y,x) = s;
			imgV.at<uchar>(y,x) = v;
		}
	}

	int delta = 10;
	int min_area = 900;
	int max_area = 40000;
	float maxVar = 1000.0;
	float minDiv = 0.5;

	vector<vector<Point> > contoursH, contoursS, contoursV;
    double t = (double)getTickCount();
	//MSER(delta, min_area, max_area, maxVar, minDiv,1,1.0,1.0,1)(imgH,contoursH);
	MSER(delta, min_area, max_area, maxVar, minDiv,1,1.0,1.0,1)(imgS,contoursS);
	MSER(delta, min_area, max_area, maxVar, minDiv,1,1.0,1.0,1)(imgV,contoursV);
	//MSER()(imgV,contoursV);
	t = (double)getTickCount() - t;
	cout<<"three calls on mser() cost "<<t*1000.0/getTickFrequency()<<"ms. "<<contoursV.size()<<endl;

	Mat ellipses;
	imgRGB.copyTo(ellipses);
    for( int i = (int)contoursS.size()-1; i >= 0; i-- )
    {
        const vector<Point>& r = contoursS[i];
		/*
        for ( int j = 0; j < (int)r.size(); j++ )
        {
            Point pt = r[j];
            img.at<Vec3b>(pt) = bcolors[i%9];
        }
		*/

        // find ellipse (it seems cvfitellipse2 have error or sth?)
        RotatedRect box = fitEllipse( r );

        box.angle=(float)CV_PI/2-box.angle;
        ellipse( ellipses, box, Scalar(196,255,255), 2 );
    }
	imwrite("result.jpg", ellipses);
	return 0;
}


int main(int argc, char* argv[])
{
	run(argv[1]);
}

