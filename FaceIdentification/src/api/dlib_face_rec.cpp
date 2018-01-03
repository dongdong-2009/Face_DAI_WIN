// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017. 
    
    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.
    
    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include "face_rec.h"
#include "libabc.h"

using namespace dlib;
using namespace std;
using namespace cv;
using namespace seeta;

#define CURRENT_VER 1
//#define _LIMIT 1

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, the jittering you can see below in jitter_image() was used during
// training, and the training dataset consisted of about 3 million images instead of 55.
// Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

frontal_face_detector detector;
shape_predictor sp;
anet_type net;
bool init_finished=0;
static FaceDetection *Seeta_detector=NULL;
static int LimitCount = 0;

static float simd(const float* x, const float* y, const long& len) {
	float sum = 0;

	for (int i = 0; i < len; i++)
	{
		sum += (x[i]-y[i]) *(x[i]-y[i]);
	}	
	return std::sqrt(sum);
}

//Function: Initialize the face detection/recognize module
//Param : 
//  ChannelNum: the max of thread
//  path: ANN binary path (can be omitted)
//Return Value:
//  0: Noraml -1: Thread Create Failed, -2: Thread Number exceed the max of thread, -3: Double Initiate, -4: No License
int Face_Rec_Init(int ChannelNum,char *path)
{
    string sp_path;
    string net_path;
    string detector_path;  
    

    int res = Check_Device_Register_State(path);

    if(res == -1 ) {
        return -4;
    }


    if(path!=NULL)
    {
        sp_path=path;
        net_path=path;
        detector_path=path;
        sp_path+="fl.dat";
        net_path+="fr_model.dat";
        detector_path+="fd.bin";
    }
    else
    {
        sp_path+="fl.dat";
        detector_path="fd.bin";
        net_path+="fr_model.dat";     
    } 

    if(Seeta_detector==NULL) {
        Seeta_detector=new FaceDetection((char *)detector_path.c_str());
        Seeta_detector->SetMinFaceSize(40);
        Seeta_detector->SetScoreThresh(3.8f);
        Seeta_detector->SetImagePyramidScaleFactor(0.8f);
        Seeta_detector->SetWindowStep(4, 4);
    }

    // The first thing we are going to do is load all our models.  First, since we need to
    // find faces in the image we will need a face detector:
    detector = get_frontal_face_detector();
    // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)

    deserialize(sp_path.c_str()) >> sp;
    // And finally we load the DNN responsible for face recognition.

    deserialize(net_path.c_str()) >> net;

    init_finished=1;	
    return 0;
}



//Function: Recognize face from picture
//Param : 
//  ChannelID: ID of the thread,
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  callback_function: Callback when complete detect

//Return Value:
//  0: Noraml, -1: Module Busy, -2: Thread Number exceed the max of thread, -3: Face Not Detected, -4: Input Paramater Null, -5 Trial Version
int Face_Rec_Extract(int ChannelID,Mat img_data_color,Mat img_data_gray,float* img_fea,Face_Rec_Extract_cb_t callback_function)
{
    int ret=0;

#ifdef _LIMIT

    struct tm *local,*ptr;
    time_t now_time; 
    now_time = time(NULL); 

    local=localtime(&now_time);
    LimitCount++;

    if ((local->tm_year > 117 && local->tm_mon > 1) || LimitCount > 1000 ) {
      cout<< "Please Use Offical Version";
      return -5;
    } else {
      cout<< "Current is trial version" << endl;
    }
#endif

    if(init_finished==0) {
        return -1;
    }
    if(img_data_color.data == NULL){
        return -4;
    }
    matrix<rgb_pixel> img;
    img.set_size(img_data_color.rows,img_data_color.cols);

    //img_data_color opencv default is BGR
    for (int n = 0; n < img_data_color.rows ;n++ )
    {
        for (int m = 0; m < img_data_color.cols; m++ )
        {
            img(n,m).red = (float)img_data_color.at<Vec3b>(n,m)[2];	//R
            img(n,m).green = (float)img_data_color.at<Vec3b>(n,m)[1];	//G
            img(n,m).blue = (float)img_data_color.at<Vec3b>(n,m)[0];	//B
        }
    }
    //pyramid_up(img);

    std::vector<matrix<rgb_pixel>> faces;

    full_object_detection shape;
    std::vector<dlib::rectangle> dets = detector(img);
    
    if(dets.size() == 0){
        dlib::rectangle det;
        det.set_left(0);
        det.set_top(0);
        det.set_right(img_data_color.cols);
        det.set_bottom(img_data_color.rows);

        shape = sp(img, det);
    }
    else{
        shape = sp(img, dets[0]);
    }

    //start 
    {
        int min_y = 999999;
        int max_y = 0;
        for (int i = 0; i < 68; i++)  
        {  
            if(shape.part(i).y()<min_y){

                min_y = shape.part(i).y();
            }
            if(shape.part(i).y()>max_y){
                max_y = shape.part(i).y();
            }
        }
                   
        if(min_y<0)
            min_y = 0;
        if(max_y<0)
            max_y = 0;
        if(max_y>img_data_color.rows)
            max_y = img_data_color.rows;

        cv::Rect Rect_Sub;
        Rect_Sub.x = 0;
        if((min_y - 10) >0 )
            Rect_Sub.y = min_y - 10;
        else
            Rect_Sub.y = min_y;
        Rect_Sub.width = img_data_color.cols;
        Rect_Sub.height = max_y - Rect_Sub.y;
        cv::Mat sub_img = img_data_color(Rect_Sub);


        matrix<rgb_pixel> img_new;
        img_new.set_size(sub_img.rows,sub_img.cols);
        //img_data_color opencv default is BGR
        for (int n = 0; n < sub_img.rows ;n++ )
        {
            for (int m = 0; m < sub_img.cols; m++ )
            {
                img_new(n,m).red = (float)sub_img.at<Vec3b>(n,m)[2]; //R
                img_new(n,m).green = (float)sub_img.at<Vec3b>(n,m)[1];   //G
                img_new(n,m).blue = (float)sub_img.at<Vec3b>(n,m)[0];    //B
            }
        }

        dlib::rectangle det1;
        det1.set_left(0);
        det1.set_top(0);
        det1.set_right(sub_img.cols);
        det1.set_bottom(sub_img.rows);
        auto shape_new = sp(img_new, det1);

        matrix<rgb_pixel> face_chip;
        extract_image_chip(img_new, get_face_chip_details(shape_new,150,0.25), face_chip);
        faces.push_back(move(face_chip));
        // Also put some boxes on the faces so we can see that the detector is finding
        // them.
        //win.add_overlay(face);

    }

    if (faces.size()<1) {
        cout<<"Did not detect face"<< endl;
        return -3;
    } else {

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  
        std::vector<matrix<float,0,1>> face_descriptors = net(faces);
    	for(int i=0;i<face_descriptors[0].size();i++)
    	{
    	    *(img_fea+i)=(float)face_descriptors[0](i,0);  //128*1
    	}
    }
    return ret;
}

//Function: Detect face from picture
//Param : 
//  ChannelID: ID of the thread,
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  callback_function: Callback when complete detect
//Return Value:
//  0: Noraml, -1: Module Busy, -2: Thread Number exceed the max of thread, -3: Face Not Detected, -4: Input Param is Null
int Face_Rec_Detect(int ChannelID,Mat img_data_color,Mat img_data_gray,std::vector<FaceInfo>&res_faces, Face_Rec_Detect_cb_t callback_function)
{
    struct  timeval start;
    struct  timeval end;
    //记录两个时间差
    unsigned  long diff;

    if(init_finished==0) {
        return -1;
    }
    if(img_data_gray.data == NULL){
        return -4;
    }

    ImageData gallery_data_gray;
    gallery_data_gray.data = img_data_gray.data;
    gallery_data_gray.width = img_data_gray.cols;
    gallery_data_gray.height = img_data_gray.rows;
    gallery_data_gray.num_channels = img_data_gray.channels();


    std::vector<seeta::FaceInfo> gallery_faces;
    gallery_faces = Seeta_detector->Detect(gallery_data_gray);
    int32_t gallery_face_num = static_cast<int32_t>(gallery_faces.size());	

    if(gallery_face_num ==0)
	   return -3;

    for(int i=0;i<gallery_face_num;i++)
    {
    	seeta::FaceInfo finalface;
    	finalface.bbox.x=(int)(gallery_faces[i].bbox.x-gallery_faces[i].bbox.width/10);
    	finalface.bbox.y=(int)(gallery_faces[i].bbox.y-gallery_faces[i].bbox.height/5);
    	finalface.bbox.width=(int)(gallery_faces[i].bbox.width+gallery_faces[i].bbox.width/5);
    	finalface.bbox.height=(int)(gallery_faces[i].bbox.height+gallery_faces[i].bbox.height/2.5);
    	if(finalface.bbox.x<0)
    	    finalface.bbox.x=0;	
    	if(finalface.bbox.y<0)
    	    finalface.bbox.y=0;
    	if(finalface.bbox.width>(gallery_data_gray.width-finalface.bbox.x))
    	    finalface.bbox.width=(gallery_data_gray.width-finalface.bbox.x);
    	if(finalface.bbox.height>(gallery_data_gray.height-finalface.bbox.y))
    	    finalface.bbox.height=(gallery_data_gray.height-finalface.bbox.y);
    	res_faces.push_back(finalface);
    }
    return 0;

#if 0
    //img_data_color opencv default is BGR
    for (int n = 0; n < img_data_color.rows ;n++ )
    {
        for (int m = 0; m < img_data_color.cols; m++ )
        {
            img(n,m).red = (float)img_data_color.at<Vec3b>(n,m)[2];	//R
            img(n,m).green = (float)img_data_color.at<Vec3b>(n,m)[1];	//G
            img(n,m).blue = (float)img_data_color.at<Vec3b>(n,m)[0];	//B
        }
    }

    std::vector<dlib::rectangle> dets = detector(img);

    if(dets.size()<1)
        return -3;

    for(int i=0;i<dets.size();i++)
    {
        dlib::FaceInfo face_infomation;
	face_infomation.bbox.x=(int)(dets[i].left())-1;
	face_infomation.bbox.y=(int)(dets[i].top())-1;
	face_infomation.bbox.width=(int)(dets[i].width())+1;
	face_infomation.bbox.height=(int)(dets[i].height())+1;
	res_faces.push_back(face_infomation);
    }
#endif
}


//Function: Compare two face feature value
//Param : 
//  img1_fea: image 1 feature value
//  img2_fea: image 2 feature value
//Return Value:
//  simularity of two faces
float Face_Rec_Compare(float * img1_fea,float * img2_fea)
{
    float leng=simd(img1_fea,img2_fea,128);
   
    if(leng>1) {
        leng=1;
    }
    else if(leng < 0){
        leng=0;
    }

    return (float)(1-leng);	
}


//Function: Deinitialize the face detection/recognize module
//Param : 
//Return Value:
//  0: Noraml
int Face_Rec_Deinit()
{
    if (Seeta_detector != NULL) {
	delete Seeta_detector;
	Seeta_detector=NULL;
    }
    return 0;
}

Face_Rec_Step_EM Face_Rec_Current_Step(int ChannelID)
{
    return FACE_REC_STEP_IDLE;
}

int Get_Face_Rec_Ver()
{
    return CURRENT_VER;
}

// ----------------------------------------------------------------------------------------

