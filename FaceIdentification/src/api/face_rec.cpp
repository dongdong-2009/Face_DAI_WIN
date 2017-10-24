#include "face_rec.h"
//#include <stdlib.h>
//#include <pthread.h>
//#include <time.h>
//#include <sys/time.h>
//#include <errno.h>
//#include <stdio.h>
//#include <unistd.h>
//#include "math_functions.h"
#include <time.h> 
#include <ctime> 

#define TEST(major, minor) major##_##minor##_Tester()
#define EXPECT_NE(a, b) if ((a) == (b)) std::cout << "ERROR: "
#define EXPECT_EQ(a, b) if ((a) != (b)) std::cout << "ERROR: "
#define _LIMIT 1

#ifdef _WIN32
std::string DATA_DIR = "../../data/";
std::string MODEL_DIR = "../../model/";
#else
std::string DATA_DIR = "../data/";
std::string MODEL_DIR = "../model/";
#endif
using namespace std;
using namespace seeta;




static FaceDetection *detector=NULL;
static FaceAlignment *point_detector=NULL;
static FaceIdentification *face_recognizer=NULL;
static int LimitCount = 0;



static float simd(const float* x, const float* y, const long& len) {

  float inner_prod = 0.0f;
  //#ifdef _WIN32
  //float op[4] = {0, 0, 0, 0};  
  __m128 X, Y; // 128-bit values
  __m128 acc = _mm_setzero_ps(); // set to (0, 0, 0, 0)
 // __m128 acc = _mm_loadu_ps(op);  
  float temp[4];

  long i;
  for (i = 0; i + 4 < len; i += 4) {
      X = _mm_loadu_ps(x + i); // load chunk of 4 floats
      Y = _mm_loadu_ps(y + i);
      acc = _mm_add_ps(acc, _mm_mul_ps(X, Y));
  }
  _mm_storeu_ps(&temp[0], acc); // store acc into an array
  inner_prod = temp[0] + temp[1] + temp[2] + temp[3];

  // add the remaining values
  for (; i < len; ++i) {
      inner_prod += x[i] * y[i];
  }
   // #endif
  return inner_prod;

}


//Function: Initialize the face detection/recognize module
//Param : 
//  path: ANN binary path (can be omitted)
//Return Value:
//  0: Noraml -1: Init Failed
int Face_Rec_Init(char *path)
{
    string alignment_path;
    string detector_path;
    string recognizer_path;   


    if(path!=NULL)
    {
        alignment_path=path;
        detector_path=path;
        recognizer_path=path;
        alignment_path+="fa.bin";
        detector_path+="fd.bin";
        recognizer_path+="fr.bin";  
    }
    else
    {
        alignment_path="fa.bin";
        detector_path="fd.bin";
        recognizer_path="fr.bin";       
    } 

    if(point_detector==NULL) {
            point_detector=new FaceAlignment((char *)alignment_path.c_str());
    }
    
    if(detector==NULL) {
        detector=new FaceDetection((char *)detector_path.c_str());
        detector->SetMinFaceSize(40);
        detector->SetScoreThresh(2.f);
        detector->SetImagePyramidScaleFactor(0.8f);
        detector->SetWindowStep(4, 4);
    }
    
    if(face_recognizer==NULL) {
        face_recognizer=new FaceIdentification((char *)recognizer_path.c_str());
    }

    return 0;
}


//Function: Recognize face from picture
//Param : 
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  img_fea: Return Feature Values

//Return Value:
//  0: Noraml, -1: Input Paramater Null, -2: Face Not Detected, -5 Trial Version
int Face_Rec_Extract(ImageData img_data_color,ImageData img_data_gray,float* img_fea)
{
    int ret=0;

#ifdef  _LIMIT 
    
    struct tm *local,*ptr;
    time_t now_time; 
    now_time = time(NULL); 

    local=localtime(&now_time);
    LimitCount++;

    if (local->tm_mon > 9 || LimitCount > 2000 ) {
      cout<< "Please Use Offical Version";
      return -5;
    } else {
      cout<< "trial version" << endl;
    }

#endif

    if((img_data_color.data == NULL)||(img_data_gray.data == NULL)|| (img_fea == NULL)) {
        return -1;
    }
          
    std::vector<seeta::FaceInfo> gallery_faces;
    gallery_faces = detector->Detect(img_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());
    if (gallery_face_num == 0) {
        return -2;
    }
    seeta::FacialLandmark gallery_points[5];
    point_detector->PointDetectLandmarks(img_data_gray, gallery_faces[0], gallery_points);   
    face_recognizer->ExtractFeatureWithCrop(img_data_color, gallery_points, img_fea);     

    return ret;
}

//Function: Detect face from picture
//Param : 
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  res_faces: return detected faces
//Return Value:
//  0: Noraml, -1: Input Param is Null, -2: Face Not Detected
int Face_Rec_Detect(ImageData img_data_color,ImageData img_data_gray,vector<FaceInfo> & res_faces)
{
    int ret=0;

    if((img_data_color.data == NULL)||(img_data_gray.data == NULL)){
        return -1;
    }

    std::vector<seeta::FaceInfo> gallery_faces;
    gallery_faces = detector->Detect(img_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());      
    if(gallery_face_num >0)
    {
        res_faces.insert(res_faces.end(),gallery_faces.begin(),gallery_faces.end());
        return 0;

    } else {

        return -2;
    }
}


//Function: Compare two face feature value
//Param : 
//  img1_fea: image 1 feature value
//  img2_fea: image 2 feature value
//Return Value:
//  simularity of two faces
float Face_Rec_Compare(float * img1_fea,float * img2_fea)
{
    long dim=2048;

    float sqr_val1 = sqrt(simd(img1_fea, img1_fea, dim));
    float sqr_val2 = sqrt(simd(img2_fea, img2_fea, dim));

    if((sqr_val1 == 0) ||(sqr_val2 == 0)) {

        return 0;        
    } else {

        return simd(img1_fea, img2_fea, dim)/(sqr_val1*sqr_val2); 
    }
}



//Function: Deinitialize the face detection/recognize module
//Param : 
//Return Value:
//  0: Noraml
int Face_Rec_Deinit()
{

    if (point_detector != NULL) {
       delete point_detector;
       point_detector=NULL;
    }

    if (detector != NULL) {
       delete detector;
       detector=NULL;
    }

    if (face_recognizer != NULL) {
       delete face_recognizer;
       face_recognizer=NULL;
    }
}

