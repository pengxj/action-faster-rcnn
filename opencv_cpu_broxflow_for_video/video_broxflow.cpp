#include <fstream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#include "of.h"
#include "COpticFlowPart.h"
//#include "io.h"

// #include "DenseTrackStab.h"
// #include "Initialize.h"
// #include "Descriptors.h"
// #include "OpticalFlowHENG.h"
#include "flo.h"
using namespace cv;
using namespace std;

static const double pi = 3.14159265358979323846;
inline static double square(int a){
    return a * a;
}

void saveOptFlowMag( Mat& flow, const string& filename, const float max_flow){
    int width = flow.cols;
    int height = flow.rows;
    Mat rgb; 
    vector<Mat> channels;
    //
    float scale = 128/max_flow;
    Mat xComp = Mat::zeros(height, width, CV_8UC1);
    Mat yComp = Mat::zeros(height, width, CV_8UC1);
    Mat xyf[2];
    split(flow, xyf);
    
    Mat mag_flow = Mat::zeros(height, width, CV_32FC1);
    xComp = xyf[0]; 
    yComp = xyf[1];
    for(int i = 0; i < xComp.rows; i++) {
        float* xc = xComp.ptr<float>(i);
        float* yc = yComp.ptr<float>(i);
        float* magc = mag_flow.ptr<float>(i);
        for(int j = 0; j < xComp.cols; j++) {
            float x = xc[j];
            float y = yc[j];
            float mag = sqrt(x*x + y*y)*scale+128;
            mag = mag<0?0:mag;
            mag = mag>255?255:mag;
            magc[j] = mag;
            // scale and center flow
            xc[j] = xc[j]*scale + 128;
            xc[j] = xc[j]<0?0:xc[j];
            xc[j] = xc[j]>255?255:xc[j];
            
            yc[j] = yc[j]*scale + 128;
            yc[j] = yc[j]<0?0:yc[j];
            yc[j] = yc[j]>255?255:yc[j];
        }
    }
    channels.push_back(mag_flow);//B
    channels.push_back(yComp);//g
    channels.push_back(xComp);//r

    merge(channels, rgb);   
    imwrite(filename, rgb); 
}


int main(int argc, char* argv[])
{
    VideoCapture capture;   
    //int show_flow = 0, save_flow = 1;
    if(argc<3){
        cout<<"usage: opticalflow [in_video_path] [out_folder]"<<endl;
        return -1;
    }
    char* video = argv[1];
    char* save_folder = argv[2];
    if (mkdir(save_folder, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
    {
        if( errno == EEXIST ) {
           // alredy exists
        }  
    }
    
    cout << "processing " << video << endl;
    capture.open(video);
    if(!capture.isOpened()) {
        cout << "Cannot open the video file" << endl;
        return -1;
    }
    int frame_num = 0;
    Mat image,  pre_image, flow;
     
    while(1){        
        Mat frame;
        // get a new frame
        capture >> frame;           
        if(frame.empty())
            break;  
        if(!frame_num){
            // initialization
            image.create(frame.size(), CV_8UC3);   
            pre_image.create(frame.size(), CV_8UC3);
            flow.create(frame.size(), CV_32FC2);
            frame.copyTo(pre_image);
            frame_num++;
            continue;
        }       
            
        frame.copyTo(image);
        
        CTensor<float> img1, img2;      
        img1.readFromOpencvMat(pre_image);
        img2.readFromOpencvMat(image);
        CTensor<float> broxflow;
        opticalFlow( img1, img2, broxflow );

        float *data = broxflow.data();
        int rows = broxflow.ySize();
        int cols = broxflow.xSize();
        // put data to Mat
        for(int i = 0; i < rows; i++) {
            float* f = flow.ptr<float>(i);
              for(int j = 0; j < cols; j++){
                  f[2*j] = data[i*cols + j];
                  f[2*j+1] = data[i*cols + j + rows*cols];
              }
          }
        saveOptFlowMag(flow, format("%s/%04d.jpg", save_folder, frame_num), 16);

        frame.copyTo(pre_image);
        frame_num++;        
        }

        capture.release();

    return 0;
}



