#include "caffe/caffe.hpp"    //为了能正常编译，需要引入caffe的头文件
#include "opencv2/core/core.hpp"            //这三行是为了引用opencv
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>                  //使用c++智能指针，必须包含该目录
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
 
int main()
{
   //初始化一个网络，网络结构从caffenet_deploy.prototxt文件中读取，TEST表示是测试阶段
   Net<float> net("caffenet_deploy.prototxt",TEST); 
   net.CopyTrainedLayersFrom("caffenet.caffemodel");   //读取已经训练好的model的参数
   vector<shared_ptr<Blob<float> > > params=net.params();    //获取网络的各个层学习到的参数(权值+偏置)
 
   //打印出该model学到的各层的参数的维度信息
   cout<<"各层参数的维度信息为：\n";
   for(int i=0;i<params.size();++i)
       cout<<params[i]->shape_string()<<endl;
   
   
    //对第一个卷积层进行可视化，第一个卷积层"conv1"的维度信息是96*3*11*11，即96个卷积核，每个卷积核是3通道的，每个卷积核尺寸为11*11	
    //故我们可以认为，该卷积层有96个图，每个图是11*11的三通道BGR图像
    int ii=0;                  //我们提前第1层的参数，此时为conv1层
    int width=params[ii]->shape(2);     //宽度,第一个卷积层为11 
    int height=params[ii]->shape(3);    //高度，第一个卷积层为11 
    int num=params[ii]->shape(0);       //卷积核的个数，第一个卷积层为96	

    //我们将num个图，放在同一张大图上进行显示，此时用OpenCV进行可视化，声明一个大尺寸的图片，使之能容纳所有的卷积核图	
    int imgHeight=(int)(1+sqrt(num))*height;  //大图的尺寸	
    int imgWidth=(int)(1+sqrt(num))*width;	
    Mat img1(imgHeight,imgWidth,CV_8UC3,Scalar(0,0,0));
     
    //同时，我们注意到各层的权值，是一个可正可负的实数，而在OpenCV里的一般图片，每个像素的值在0~255之间	
    //因为我们还需要对权值进行归一化到0~255才能正常显示
    float maxValue=-1000,minValue=10000;	
    const float* tmpValue=params[ii]->cpu_data();   //获取该层的参数，实际上是一个一维数组	
    for(int i=0;i<params[ii]->count();i++)
    {        //求出最大最小值		
        maxValue=std::max(maxValue,tmpValue[i]);		
        minValue=std::min(minValue,tmpValue[i]);	
    }
    //对最终显示的大尺寸图片，进行逐个像素赋值
    int kk=0;                         //此时在画第kk个卷积核
    for(int y=0;y<imgHeight;y+=height)
    {		
        for(int x=0;x<imgWidth;x+=width)
        {			
            if(kk>=num)				
            continue;
            Mat roi=img1(Rect(x,y,width,height));			
            for(int i=0;i<height;i++)
            {
                for(int j=0;j<width;j++)
                {					
                    for(int k=0;k<3;k++)
                    {					
                        float value=params[ii]->data_at(kk,k,i,j);    
                        roi.at<Vec3b>(i,j)[k]=(value-minValue)/(maxValue-minValue)*255;   //归一化到0~255
                    }
                }
             }
            ++kk;
        }
    }
    resize(img1,img1,Size(500,500));   //将显示的大图，调整为500*500尺寸	
    imwrite("conv1",img1);              //显示	
    //waitKey(0); 
   return 0;
}
