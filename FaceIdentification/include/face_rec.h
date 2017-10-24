
//#include <iostream>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
//#include "math_functions.h"

using namespace seeta;
using namespace std;


int Face_Rec_Init(char *path);
int Face_Rec_Detect(ImageData img_data_color,ImageData img_data_gray,vector<FaceInfo> & res_faces);
int Face_Rec_Extract(ImageData img_data_color,ImageData img_data_gray,float* img_fea);
float Face_Rec_Compare(float * img1_fea,float * img2_fea);
int Face_Rec_Deinit();


