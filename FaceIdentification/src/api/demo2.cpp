#include "face_rec.h"
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include <conio.h>

//#include "opencv2/core/version.hpp"
#include "opencv/cv.h"
#include "opencv/highgui.h"

using namespace std;
using namespace seeta;
using namespace cv;

float* gallery_fea = NULL;
float* probe_fea = NULL;

ImageData gallery_src_data_color;
ImageData gallery_src_data_gray;
ImageData gallery_dst_data_color;
ImageData gallery_dst_data_gray;



int main(int argc, char *argv[]) 
{
	char* srcImg=argv[1];
	char* dstImg=argv[2];
	std::vector<seeta::FaceInfo> ga_faces;
	// Image Data Preparation

	cv::Mat gallery_img_color = cv::imread(srcImg, 1);
	cv::Mat gallery_img_gray;
	cv::cvtColor(gallery_img_color, gallery_img_gray, CV_BGR2GRAY);

	gallery_src_data_color.data = gallery_img_color.data;
    gallery_src_data_color.width = gallery_img_color.cols;
    gallery_src_data_color.height = gallery_img_color.rows;
    gallery_src_data_color.num_channels = gallery_img_color.channels();

	gallery_src_data_gray.data = gallery_img_gray.data;
    gallery_src_data_gray.width = gallery_img_gray.cols;
    gallery_src_data_gray.height = gallery_img_gray.rows;
    gallery_src_data_gray.num_channels = gallery_img_gray.channels();

	cv::Mat probe_img_color = cv::imread(dstImg, 1);
	cv::Mat probe_img_gray;
	cv::cvtColor(probe_img_color, probe_img_gray, CV_BGR2GRAY);

	gallery_dst_data_color.data = probe_img_color.data;
    gallery_dst_data_color.width = probe_img_color.cols;
    gallery_dst_data_color.height = probe_img_color.rows;
    gallery_dst_data_color.num_channels = probe_img_color.channels();

	gallery_dst_data_gray.data = probe_img_gray.data;
    gallery_dst_data_gray.width = probe_img_gray.cols;
    gallery_dst_data_gray.height = probe_img_gray.rows;
    gallery_dst_data_gray.num_channels = probe_img_gray.channels();

	
	gallery_fea = new float[2048];           
	probe_fea = new float[2048]; 

	// Using FaceDAI Library

	Face_Rec_Init(NULL);
	
	Face_Rec_Extract(gallery_src_data_color,gallery_src_data_gray,gallery_fea);
	Face_Rec_Extract(gallery_dst_data_color,gallery_dst_data_gray,probe_fea);
	ga_faces.clear();
	Face_Rec_Detect(gallery_src_data_color,gallery_src_data_gray,ga_faces);
	std::cout << "picture 1 detect faces:"<<"face num:"<<ga_faces.size()<< endl;
	ga_faces.clear();
	Face_Rec_Detect(gallery_dst_data_color,gallery_dst_data_gray,ga_faces);
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
