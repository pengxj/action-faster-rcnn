#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"
#include "flo.h"
using namespace cv;
using namespace std;

static const double pi = 3.14159265358979323846;
inline static double square(int a){
	return a * a;
}

// draw optical flow as line and circle
void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {  
 for(int y = 0; y < cflowmap.rows; y += step)  
        for(int x = 0; x < cflowmap.cols; x += step){  
            const Point2f& fxy = flow.at< Point2f>(y, x);  
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),  
                 color);  
            circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 2, color, -1);  
        }  
} 

void drawOptFlowMapArrow (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
	int line_thickness = 2;
 	for(int y = 0; y < cflowmap.rows; y += step)  
        for(int x = 0; x < cflowmap.cols; x += step){  
            const Point2f& fxy = flow.at< Point2f>(y, x);
            Point p,q;
			p.x = x;
			p.y = y;
			q.x = cvRound(x+fxy.x);
			q.y = cvRound(y+fxy.y);
            double angle;
			double hypotenuse;
			angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
			hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) );
			/* Here we lengthen the arrow by a factor of three. */
			q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
			q.y = (int) (p.y - 3 * hypotenuse * sin(angle));
			
            line(cflowmap, p, q, color,line_thickness);  
			p.x = (int) (q.x + 9 * cos(angle + pi / 4));
			p.y = (int) (q.y + 9 * sin(angle + pi / 4));
			line( cflowmap, p, q, color,line_thickness);
			
			p.x = (int) (q.x + 9 * cos(angle - pi / 4));
			p.y = (int) (q.y + 9 * sin(angle - pi / 4));
			line( cflowmap, p, q, color,line_thickness);            
        }  
} 
void drawOptFlowMag (const Mat& flow, bool bcolor){
	
	int width = flow.cols;
	int height = flow.rows;
//	Mat xComp = Mat::zeros(height, width, CV_32FC1);
//	Mat yComp = Mat::zeros(height, width, CV_32FC1);
//	Mat hsv = Mat::zeros(height, width, CV_8UC3);
//	Mat channels[3];
	Mat rgb = Mat::zeros(height, width, CV_8UC3);
	IplImage ipl_flow = flow;
	IplImage ipl_rgb = rgb;
	MotionToColor(&ipl_flow, &ipl_rgb);
	imshow("magnitudeColor", rgb);
    waitKey(5);
//	if(bcolor){		
//		split(hsv, channels);
//		channels[1] = Mat(height, width, CV_8UC1, Scalar(255));
//	}
//	for(int i = 0; i < height; i++) {
//		const float* f = flow.ptr<float>(i);
//		float* xf = xComp.ptr<float>(i);
//		float* yf = yComp.ptr<float>(i);
//		for(int j = 0; j < width; j++) {
//			xf[j] = f[2*j];
//			yf[j] = f[2*j+1];
//		}
//	}
//	Mat magnitude, angle;
//    cartToPolar(xComp, yComp, magnitude, angle);
//    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);
//    if(bcolor){
//    	angle = angle*180/pi/2;
//    	angle.convertTo(angle, CV_8UC1);
//		// set angle to 0 channel of hsv
//		channels[0] = angle;
//    	// set magnitude to 2 channel of hsv
//    	channels[2] = magnitude;
//    	    	
//    	merge(channels, 3, hsv);
//    	cvtColor(hsv, rgb, CV_HSV2BGR);
//    	imshow("magnitudeColor", rgb);
//    	waitKey(5);
//    }
//    else{
//    imshow("magnitude", magnitude);
//    waitKey(5);
//    }
}
void drawOptFlowChs(const Mat& flow){
	int width = flow.cols;
	int height = flow.rows;
	Mat xComp = Mat::zeros(height, width, CV_8UC1);
	Mat yComp = Mat::zeros(height, width, CV_8UC1);
	Mat xyf[2];
	split(flow, xyf);
	normalize(xyf[0], xComp, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(xyf[1], yComp, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("Horizontal", xComp);
	imshow("Vertical", yComp);
}
// save flow as two uint8 image
void saveOptFlowXY(const Mat& flow, const string& filenameX, const string& filenameY){
	int width = flow.cols;
	int height = flow.rows;
	Mat xComp = Mat::zeros(height, width, CV_8UC1);
	Mat yComp = Mat::zeros(height, width, CV_8UC1);
	Mat xyf[2];
	split(flow, xyf);
	normalize(xyf[0], xComp, 0, 255, NORM_MINMAX, CV_8UC1);
	normalize(xyf[1], yComp, 0, 255, NORM_MINMAX, CV_8UC1);
	imwrite(filenameX, xComp);
	imwrite(filenameY, yComp);
}
void saveOptFlowMag(const Mat& flow, const string& filename, const float maxMotion){
	int width = flow.cols;
	int height = flow.rows;
	Mat rgb = Mat::zeros(height, width, CV_8UC3);
	IplImage ipl_flow = flow;
	IplImage ipl_rgb = rgb;
	MotionToColor(&ipl_flow, &ipl_rgb, maxMotion);
	imwrite(filename, rgb);
}
void MedianBlurFlow(Mat& flow, const int ksize, const bool bmeanSub = 0)
{
	int width = flow.cols;
	int height = flow.rows;

	Mat flowX = Mat::zeros(height, width, CV_32FC1);
	Mat flowY = Mat::zeros(height, width, CV_32FC1);

	for(int i = 0; i < height; i++) {
		const float* f = flow.ptr<float>(i);
		float* fx = flowX.ptr<float>(i);
		float* fy = flowY.ptr<float>(i);
		for(int j = 0; j < width; j++) {
			fx[j] = f[2*j];
			fy[j] = f[2*j+1];
		}
	}

	medianBlur(flowX, flowX, ksize);
	medianBlur(flowY, flowY, ksize);
	if(bmeanSub){
		flowX = flowX - mean(flowX)[0];
		flowY = flowY - mean(flowY)[0];
	}
	for(int i = 0; i < height; i++) {
		float* f = flow.ptr<float>(i);
		const float* fx = flowX.ptr<float>(i);
		const float* fy = flowY.ptr<float>(i);
		for(int j = 0; j < width; j++) {
			f[2*j] = fx[j];
			f[2*j+1] = fy[j];
		}
	}
}
int main(int argc, char* argv[])
{
	VideoCapture capture;	
	int show_flow = 0, save_flow = 1;
	if(argc<4){
		cout<<"usage: opticalflow [root_videofolder] [root_saveOFfolder] [video_file.txt] [0|1:bStab]"<<endl;
		return -1;
	}
	char* video_folder = argv[1];
	char* save_folder = argv[2];
	char* video_listfile = argv[3];
	int bStab = atoi(argv[4]);
	cout << bStab << endl;
	// read the video list file
	ifstream lf_istream(video_listfile);
	if(!lf_istream.is_open())
         cout<<"Cannot open listfile"<<endl;
    int nfile = 0;
    while(!lf_istream.eof()){
	// loop for videos
		string video;
		lf_istream >> video;
		nfile++;
		cout << nfile << "processing " << video << endl;
		capture.open(string(video_folder) + "/" + video);
		if(!capture.isOpened()) {
			cout << "Cannot open the video file" << endl;
			return -1;
		}
		int frame_num = 0;
		Mat image, prev_grey, grey, flow;
		SurfFeatureDetector detector_surf(200);
		SurfDescriptorExtractor extractor_surf(true, true);

		std::vector<Point2f> prev_pts_flow, pts_flow;
		std::vector<Point2f> prev_pts_surf, pts_surf;
		std::vector<Point2f> prev_pts_all, pts_all;

		std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
		Mat prev_desc_surf, desc_surf;
		Mat human_mask;
	
		while(1){        
			Mat frame;
			// get a new frame
			capture >> frame;			
			if(frame.empty())
				break;		
			// skip every second frame
//			if(frame_num % 2 == 1) {
//				frame_num++;
//				continue;
//			}
			if(!frame_num){
				// initialization
				image.create(frame.size(), CV_8UC3);
				grey.create(frame.size(), CV_8UC1);
				prev_grey.create(frame.size(), CV_8UC1);
				cvtColor(frame, prev_grey, CV_BGR2GRAY);
				
				human_mask = Mat::ones(frame.size(), CV_8UC1);
				detector_surf.detect(prev_grey, prev_kpts_surf, human_mask);
				extractor_surf.compute(prev_grey, prev_kpts_surf, prev_desc_surf);
			
				frame.copyTo(image);
				frame_num++;
				continue;
			}		
			
			frame.copyTo(image);
			cvtColor(image, grey, CV_BGR2GRAY);
			if(bStab){
				detector_surf.detect(grey, kpts_surf, human_mask);
				extractor_surf.compute(grey, kpts_surf, desc_surf);
				// surf point matching
				ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
			
				calcOpticalFlowFarneback(prev_grey, grey, flow, sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
				MedianBlurFlow(flow, 3, 0);		
				MatchFromFlow(prev_grey, flow, prev_pts_flow, pts_flow, human_mask);		
				MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

				Mat H = Mat::eye(3, 3, CV_64FC1);
				if(pts_all.size() > 50) {
					std::vector<unsigned char> match_mask;
					Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
					if(countNonZero(Mat(match_mask)) > 25)
						H = temp;
					//cout << H << endl;
				}
				Mat H_inv = H.inv();
				Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
				MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame
				if(show_flow){
					Mat Vcon;
					vconcat(prev_grey, grey_warp,Vcon);
					imshow("GreyWarp", Vcon);
				}
				calcOpticalFlowFarneback(prev_grey, grey_warp, flow,sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
				MedianBlurFlow(flow, 3, 0);
				prev_kpts_surf = kpts_surf;
				desc_surf.copyTo(prev_desc_surf);
			}
			else{
				calcOpticalFlowFarneback(prev_grey, grey, flow,sqrt(2)/2.0, 5, 10, 2, 7, 1.5, OPTFLOW_FARNEBACK_GAUSSIAN);
				MedianBlurFlow(flow, 3);
			}
			// save optical flow as images
			if(save_flow){
				string strfolder = format("%s/%s.flow", save_folder, video.c_str());
//				saveOptFlowXY(flow, format("%s/%d_x.jpg", strfolder.c_str(), frame_num), format("%s/%d_y.jpg", strfolder.c_str(), frame_num));
				saveOptFlowMag(flow, format("%s/%04d.jpg", strfolder.c_str(), frame_num), 30);
			}
			
			if(show_flow) {
//				drawOptFlowMap(flow, image, 10, CV_RGB(0, 255, 0));	
				//	drawOptFlowMapArrow(flow, image, 5, CV_RGB(0, 255, 0));		
					drawOptFlowMag(flow, true);     
				//	drawOptFlowChs(flow); 
				imshow("OpticalFlowFarneback", image); 
			}
			
			grey.copyTo(prev_grey);
			frame_num++;
		    if(waitKey(10) % 0x100 == 27)
				break; 
		}
		capture.release();
	}
    return 0;
}



