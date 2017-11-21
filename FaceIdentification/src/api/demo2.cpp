#include "face_rec.h"
#include <iostream>
#include <fstream>
#include <sstream>
//#include <conio.h>
#include <sys/time.h>
//#include <io.h>


#ifdef DUSE_DLIB
#define FEATURE_LENGTH 128
using namespace seeta;
using namespace dlib;
#else
#define FEATURE_LENGTH 2048
using namespace seeta;
#endif

using namespace std;
using namespace cv;

float* gallery_fea = NULL;
float* probe_fea = NULL;


int main(int argc, char *argv[]) 
{
    /*
	struct  timeval start;
    struct  timeval end;
    unsigned  long diff;
	*/

	char* srcImg=argv[1];
	char* dstImg=argv[2];
	std::vector<FaceInfo> ga_faces;
	// Image Data Preparation

	cv::Mat gallery_src_data_color = cv::imread(srcImg, 1);
	cv::Mat gallery_src_data_gray;
	cv::cvtColor(gallery_src_data_color, gallery_src_data_gray, CV_BGR2GRAY);


	cv::Mat probe_dst_data_color = cv::imread(dstImg, 1);
	cv::Mat probe_dst_data_gray;
	cv::cvtColor(probe_dst_data_color, probe_dst_data_gray, CV_BGR2GRAY);


	
	gallery_fea = new float[FEATURE_LENGTH];           
	probe_fea = new float[FEATURE_LENGTH]; 

	// Using FaceDAI Library

	cout<<"start init"<<endl;
	Face_Rec_Init(1);
	

	cout<<"start extract"<<endl;


//	gettimeofday(&start,NULL);
	Face_Rec_Extract(0,gallery_src_data_color,gallery_src_data_gray,gallery_fea,NULL);
/*
    gettimeofday(&end,NULL);
    diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	cout<<"first extract time is :"<<(diff/1000000.0)<<endl;


	gettimeofday(&start,NULL);
*/

	Face_Rec_Extract(0,probe_dst_data_color,probe_dst_data_gray,probe_fea,NULL);
/*
    gettimeofday(&end,NULL);
    
    diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	cout<<"second extract time is :"<<(diff/1000000.0)<<endl;
*/
	cout<<"start Detect"<<endl;

	ga_faces.clear();

//	gettimeofday(&start,NULL);
	Face_Rec_Detect(0,gallery_src_data_color,gallery_src_data_gray,ga_faces,NULL);

/*
        gettimeofday(&end,NULL);
  
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	cout<<"first Detect time is :"<<(diff/1000000.0)<<endl;
*/
	std::cout << "picture 1 detect faces:"<<"face num:"<<ga_faces.size()<< endl;
	ga_faces.clear();


//	gettimeofday(&start,NULL);
	Face_Rec_Detect(0,probe_dst_data_color,probe_dst_data_gray,ga_faces,NULL);

/*
        gettimeofday(&end,NULL);
        diff = 1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	cout<<"first Detect time is :"<<(diff/1000000.0)<<endl;
*/
	std::cout << "picture 2 detect faces:"<<"face num:"<<ga_faces.size()<< endl;



	//Caculate Sim
	float sim = Face_Rec_Compare(gallery_fea,probe_fea);
	std::cout <<"sim of two face is:"<< sim <<endl;


	Face_Rec_Deinit();
	//std::cout << "two picture detect successfully"<<endl;
	//std::cout << "demo is over, press any key to exit!!!"<<endl;
	
	delete(gallery_fea);
	delete(probe_fea);
	gallery_fea=NULL;
	probe_fea=NULL;	
	
	return 0;
}
