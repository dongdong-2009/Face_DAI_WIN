#include "face_rec.h"
#include <iostream>
#include <fstream>
#include <sstream>
//#include <conio.h>
#include <sys/time.h>

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
	int retFlag = 0;
	ga_faces.clear();
	retFlag = Face_Rec_Detect(0,gallery_src_data_color,gallery_src_data_gray,ga_faces,NULL);
	if(retFlag !=0){
		std::cout <<"sFace_Rec_Detect fail"<<endl;
		return 0;
	}

    cv::Rect face_rect;
    face_rect.x = ga_faces[0].bbox.x;
    face_rect.y = ga_faces[0].bbox.y;
    face_rect.width = ga_faces[0].bbox.width;
    face_rect.height = ga_faces[0].bbox.height;
    cv::Mat face_mat = gallery_src_data_color(face_rect);
    cv::imshow("1", gallery_src_data_color);
    //cv::imshow("1-seeta", face_mat);

	int res = Face_Rec_Extract(0,face_mat,gallery_src_data_gray,gallery_fea,NULL);
	cout << "Res1 of Face_Rec_Extract"<<res <<endl;

	ga_faces.clear();
	retFlag = Face_Rec_Detect(0,probe_dst_data_color,probe_dst_data_gray,ga_faces,NULL);
	if(retFlag !=0){
		std::cout <<"sFace_Rec_Detect fail"<<endl;
		return 0;
	}
	
    cv::Rect face_rect1;
    face_rect1.x = ga_faces[0].bbox.x;
    face_rect1.y = ga_faces[0].bbox.y;
    face_rect1.width = ga_faces[0].bbox.width;
    face_rect1.height = ga_faces[0].bbox.height;
    cv::Mat face_mat1 = probe_dst_data_color(face_rect1);
    cv::imshow("2", probe_dst_data_color);
    //cv::imshow("2-seeta", face_mat1);

	res = Face_Rec_Extract(0,face_mat1,probe_dst_data_gray,probe_fea,NULL);
	cout << "Res2 of Face_Rec_Extract"<<res <<endl;

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
	
	cv::waitKey();

	return 0;
}
