
#include <iostream>
#include "face_identification.h"
#include "recognizer.h"
#include "face_detection.h"
#include "face_alignment.h"
#include "math_functions.h"
#include "opencv2/core/version.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#ifdef DUSE_DLIB
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#endif

using namespace std;
using namespace cv;



#ifdef DUSE_DLIB
using namespace dlib;
#endif
using namespace seeta;


#define Face_Rec_Pthread_MAX_NUM    64

typedef void(*Face_Rec_Extract_cb_t)(int state,int FaceNum,float* img_fea);
typedef void(*Face_Rec_Detect_cb_t)(int state,int FaceNum,void * face_data);
typedef enum { /* bitmapped status flags */
    FACE_REC_STEP_IDLE,
    FACE_REC_STEP_EXTR,
    FACE_REC_STEP_DECT,    
    FACE_REC_STEP_COMP
} Face_Rec_Step_EM;

int Face_Rec_Init(int ChannelNum,char *path = NULL);
int Face_Rec_Detect(int ChannelID,Mat img_data_color,Mat img_data_gray,std::vector<FaceInfo> & res_faces, Face_Rec_Detect_cb_t callback_function);
int Face_Rec_Extract(int ChannelID,Mat img_data_color,Mat img_data_gray,float* img_fea,Face_Rec_Extract_cb_t callback_function);			
float Face_Rec_Compare(float * img1_fea,float * img2_fea);
int Face_Rec_Deinit();
Face_Rec_Step_EM Face_Rec_Current_Step(int ChannelID);
// return value  -2:id error    -1: other error


