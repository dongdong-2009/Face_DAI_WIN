FaceDAI Library API Usage
Definition of details can be referred at face_rec.h


//Function: Initialize the face detection/recognize module
//Param : 
//  path: ANN binary path (can be omitted)
//Return Value:
//  0: Noraml -1: Init Failed
int Face_Rec_Init(char *path)


//Function: Recognize face from picture
//Param : 
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  img_fea: Return Feature Values

//Return Value:
//  0: Noraml, -1: Input Paramater Null, -2: Face Not Detected, 
int Face_Rec_Extract(ImageData img_data_color,ImageData img_data_gray,float* img_fea)



//Function: Detect face from picture
//Param : 
//  img_data_color: Original Image,
//  img_data_gray: Gray Image,
//  res_faces: return detected faces
//Return Value:
//  0: Noraml, -1: Input Param is Null, -2: Face Not Detected
int Face_Rec_Detect(ImageData img_data_color,ImageData img_data_gray,vector<FaceInfo> & res_faces)


//Function: Compare two face feature value
//Param : 
//  img1_fea: image 1 feature value
//  img2_fea: image 2 feature value
//Return Value:
//  simularity of two faces
float Face_Rec_Compare(float * img1_fea,float * img2_fea)


//Function: Deinitialize the face detection/recognize module
//Param : 
//Return Value:
//  0: Noraml
int Face_Rec_Deinit()