#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include "libabc.h"

using namespace std;

int main( int argc, char **argv )
{
    int len=20;	
    char *inBuffer=(char *)malloc(len+1); //输入缓冲区
    int num=Get_GUID_For_Device(inBuffer);	
    cout<<"GUID for this device is: "<<inBuffer<<"   GUID num is: "<<num<<endl;

    free(inBuffer);
    inBuffer = NULL;
	
    if(argc == 2)
    {
	char * temp=argv[1]; 
	if(strlen(temp)==60)
	{
	    Register_For_Device(temp);
	}
	else
	{
	    printf("Parameter error!!!\n");
	    return -1;
	}	
    }
    if(0 == Check_Device_Register_State())
    	cout<<"This device had allready registered"<<endl;
    else
	cout<<"This device did not register,Please contact us"<<endl;
    return 0;
}
