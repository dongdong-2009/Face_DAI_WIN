#include "face_rec.h"
#include <iostream>
#include <fstream>
#include <sstream>
//#include <conio.h>
#include <sys/time.h>
#include <io.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>

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
float* gallery_fea_pair = NULL;
float* probe_fea_pair = NULL;
std::vector<string> pic_name;
std::vector<string> id_file_whole_name;
std::vector<string> photo_file_whole_name;

void listline(const char * filename,const char * dirc);
void listFiles(const char * filename);
string compare_function(string file1,string file2);
int pic_num=0;
int single_pic=0;
FILE *file_stream_unpair = NULL;
FILE *file_stream_pair = NULL;
string name1_backup="";

int main(int argc, char *argv[]) 
{
	gallery_fea = new float[FEATURE_LENGTH];           
	probe_fea = new float[FEATURE_LENGTH]; 
	gallery_fea_pair = new float[FEATURE_LENGTH];           
	probe_fea_pair = new float[FEATURE_LENGTH]; 
	
	// Using FaceDAI Library

	cout<<"start init"<<endl;
	Face_Rec_Init(1);
	

	cout<<"start extract"<<endl;

	file_stream_unpair = fopen("not_pair_face_result.txt", "w+");
	file_stream_pair = fopen("pair_face_result.txt", "w+");
	
	char* name=argv[1];
	char* dir=argv[2];
	single_pic=0;
	
    struct stat stat_file;
    if(stat(name, &stat_file) == 0){
        if((stat_file.st_mode & S_IFMT) == S_IFDIR){
			id_file_whole_name.clear();
			photo_file_whole_name.clear();
            listFiles(name);
			
			for(int p=0;p<id_file_whole_name.size();p++)
			{
				for(int q=0;q<photo_file_whole_name.size();q++)
				{
					if(p!=q)
					{
						cout << id_file_whole_name.at(p)<<"  with  "<<photo_file_whole_name.at(q)<< "\n";			
						string strSave = compare_function(id_file_whole_name.at(p),photo_file_whole_name.at(q));
						//strSave+=" is the sim of ";
						//strSave+=pic_name.at(0);
						//strSave+=" and ";		
						//strSave+=pic_name.at(1);
						strSave+="\n";
						if(NULL != file_stream_unpair)
							fwrite(strSave.c_str(), sizeof(char),strSave.length() ,file_stream_unpair);		
					}
				}
			}		
        }
		else
		{
			listline(name,dir);
		}
    }else{
		cout<<"parameter should be: dir  or ***.txt and dir" <<endl;
    }	
	
	if(NULL != file_stream_unpair)
		fclose(file_stream_unpair);
	if(NULL != file_stream_pair)
		fclose(file_stream_pair);	
	
	Face_Rec_Deinit();
	//std::cout << "two picture detect successfully"<<endl;
	//std::cout << "demo is over, press any key to exit!!!"<<endl;

	
	delete(gallery_fea);
	delete(probe_fea);
	delete(gallery_fea_pair);
	delete(probe_fea_pair);
	
	gallery_fea=NULL;
	probe_fea=NULL;	
	gallery_fea_pair=NULL;
	probe_fea_pair=NULL;	
	
	return 0;
}

void listFiles(const char * dir)
{
    char dirNew[200];
	char dirNew_Forload[200];
    strcpy(dirNew, dir);
	strcpy(dirNew_Forload, dir);
    strcat(dirNew, "\\*.*");    // 在目录后面加上"\\*.*"进行第一次搜索

    intptr_t handle;
    _finddata_t findData;

    handle = _findfirst(dirNew, &findData);
    if (handle == -1)        // 检查是否成功
        return;

    do
    {
        if (findData.attrib & _A_SUBDIR)
        {
            if (strcmp(findData.name, ".") == 0 || strcmp(findData.name, "..") == 0)
                continue;

            //cout << findData.name << "\t<dir>\n";

            // 在目录后面加上"\\"和搜索到的目录名进行下一次搜索
            strcpy(dirNew, dir);
            strcat(dirNew, "\\");
            strcat(dirNew, findData.name);
			pic_num=0;
            listFiles(dirNew);
        }
        else
		{
			//cout << findData.name << "\t" << findData.size << " bytes.\n";
			string name=findData.name;
			pic_name.push_back(name);
			pic_num++;
		}
    } while (_findnext(handle, &findData) == 0);
	
	if(pic_num>=2)
	{
		string file_name1=dirNew_Forload;
		string file_name2=dirNew_Forload;
		file_name1=file_name1+"\\"+pic_name.at(0);
		file_name2=file_name2+"\\"+pic_name.at(1);

		cout << file_name1<<"  with  "<<file_name2<< "\n";
	
		id_file_whole_name.push_back(file_name1);
		photo_file_whole_name.push_back(file_name2);
		
		string strSave = compare_function(file_name1,file_name2);
		//strSave+=" is the sim of ";
		//strSave+=pic_name.at(0);
		//strSave+=" and ";		
		//strSave+=pic_name.at(1);
		strSave+="\n";
		if(NULL != file_stream_pair)
			fwrite(strSave.c_str(), sizeof(char),strSave.length() ,file_stream_pair);
	}	
	else if(pic_num==1)
	{
		string file_name=dirNew_Forload;
		file_name=file_name+"\\"+pic_name.at(0);

		if(single_pic==0)
		{
			single_pic=1;
			cv::Mat gallery_src_data_color = cv::imread(file_name, 1);
			cv::Mat gallery_src_data_gray;
			cv::cvtColor(gallery_src_data_color, gallery_src_data_gray, CV_BGR2GRAY);			
			Face_Rec_Extract(0,gallery_src_data_color,gallery_src_data_gray,gallery_fea,NULL);

			name1_backup=pic_name.at(0);
			cout << file_name<<"  step 111  "<< "\n";	
		}
		else	
		{
			single_pic=0;
			cv::Mat probe_dst_data_color = cv::imread(file_name, 1);
			cv::Mat probe_dst_data_gray;
			cv::cvtColor(probe_dst_data_color, probe_dst_data_gray, CV_BGR2GRAY);
			Face_Rec_Extract(0,probe_dst_data_color,probe_dst_data_gray,probe_fea,NULL);
			float sim = Face_Rec_Compare(gallery_fea,probe_fea);
			ostringstream buffer;
			buffer << sim;
			string str = buffer.str();				
			
			string strSave=str;
			//strSave+=" is the sim of ";
			//strSave+=name1_backup;
			//strSave+=" and ";		
			//strSave+=pic_name.at(0);
			strSave+="\n";
			if(NULL != file_stream_unpair)
				fwrite(strSave.c_str(), sizeof(char),strSave.length() ,file_stream_unpair);					
			cout << file_name<<"  step 222  "<< "\n";			
		}
	}
	pic_num=0;	
	pic_name.clear();
    _findclose(handle);    // 关闭搜索句柄
}

void listline(const char * filename,const char * dirc)
{
	FILE *pair_txt = fopen(filename, "rb");
	if(!pair_txt){
        cout << " read file error  "<< "\n";
        return;
    }
    fseek(pair_txt, 0, SEEK_END);
    int nleft = ftell(pair_txt);
    fseek(pair_txt, 0, SEEK_SET);
	
    char buff[100] = {0x00};
    int nread = 0;//, nleft = stat_file.st_size;
	while (fgets(buff, 100, pair_txt) != NULL)
	{
		string str=buff;
		nread=str.size();
		nleft -= (nread+1);
		
		string name1="";
		string name2="";
		string num1="";
		string num2="";		
		string final_file1="";
		string final_file2="";			
		size_t posz1,posz2,posz3,posz4;
		string skip="	";
		bool isoneman=0;
		
		if((posz1=str.find(skip,0))!=string::npos)
			name1=str.substr(0,posz1);
		if((posz2=str.find(skip,posz1+skip.length()))!=string::npos)
			num1=str.substr(posz1+skip.length(),posz2-(posz1+skip.length()));
		if((posz3=str.find(skip,posz2+skip.length()))!=string::npos)
		{
			name2=str.substr(posz2+skip.length(),posz3-(posz2+skip.length()));
			if(str.at(nread-1)=='\n')
				num2=str.substr(posz3+skip.length(),nread-posz3-2);
			else
				num2=str.substr(posz3+skip.length(),nread-posz3);
		}	
		else
		{
			isoneman=1;
			name2=name1;
			if(str.at(nread-1)=='\n')
				num2=str.substr(posz2+skip.length(),nread-posz2-2);
			else
				num2=str.substr(posz2+skip.length(),nread-posz2);
		}	

		if(num1.length()==1)
			num1="00"+num1;
		else if(num1.length()==2)
			num1="0"+num1;		
		if(num2.length()==1)
			num2="00"+num2;
		else if(num2.length()==2)
			num2="0"+num2;
		
		string dir=dirc;
		string filename11=name1+"_0"+num1+".jpg";
		string filename22=name2+"_0"+num2+".jpg";
		final_file1=dir+"\\"+name1+"\\"+filename11;
		final_file2=dir+"\\"+name2+"\\"+filename22;
		cout<<final_file1<<" compare with "<<final_file2;
		string strSave=compare_function(final_file1,final_file2);
		cout<<" result is "<<strSave<<"\n";

		strSave+="\n";
		if(isoneman)
		{
			if(NULL != file_stream_pair)
				fwrite(strSave.c_str(), sizeof(char),strSave.length() ,file_stream_pair);	
		}
		else
		{
			if(NULL != file_stream_unpair)
				fwrite(strSave.c_str(), sizeof(char),strSave.length() ,file_stream_unpair);					
		}
	}
    fclose(pair_txt);	
}

string compare_function(string file1,string file2)
{
	cv::Mat gallery_src_data_color = cv::imread(file1, 1);
	cv::Mat gallery_src_data_gray;
	cv::cvtColor(gallery_src_data_color, gallery_src_data_gray, CV_BGR2GRAY);
	
	cv::Mat probe_dst_data_color = cv::imread(file2, 1);
	cv::Mat probe_dst_data_gray;
	cv::cvtColor(probe_dst_data_color, probe_dst_data_gray, CV_BGR2GRAY);	
	
	Face_Rec_Extract(0,gallery_src_data_color,gallery_src_data_gray,gallery_fea_pair,NULL);
	Face_Rec_Extract(0,probe_dst_data_color,probe_dst_data_gray,probe_fea_pair,NULL);
	float sim = Face_Rec_Compare(gallery_fea_pair,probe_fea_pair);	
	
	ostringstream buffer;
	buffer << sim;
	string str = buffer.str();	
	return str;
}