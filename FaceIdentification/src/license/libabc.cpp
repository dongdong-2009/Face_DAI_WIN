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
#include <unistd.h>
#include "libabc.h"
#include <sys/system_properties.h>

using namespace std;

int Key_n=31571;
int Key_d=15031;
int GUID_LENGTH=12;

void rsa_decrypt( int n, int d, int *&cw, int cLength, char *mw );

void rsa_decrypt( int n, int d, int *&cw, int cLength, char *mw ) {

	int i=0, j=-1;
	long temInt = 0;

	for( i=0; i<cLength/5; ++i ) {

		mw[i] = 0;
		temInt = cw[i];

		if( d != 0 ) {
			
			for( j=1; j<d; j++ ) {
				
				temInt = (long)( temInt * cw[i] ) % n;
			}
		} else {

			temInt = 1;
		}

		mw[i] = (char)temInt;
	}
}

int Get_GUID_For_Device(char * GUID) { 

	int len1=17;
	int len2=12;
	int id_len1,id_len2;

	FILE *stream;
	char *temp1=(char *)malloc(len1+10); 
	char *temp2=(char *)malloc(len2+10);
	memset(temp1,'\0',len1+10);
	memset(temp2,'\0',len2+10);
	stream = popen( "cat /sys/class/net/wlan0/address", "r" ); 
	int num=fread( temp1, sizeof(char), len1+10, stream); 

	id_len1=0;
	num=num-1;
	for(int i=0;i<num;i++) {

	    if(temp1[i]!=':') {

	    	*(temp2+id_len1)=temp1[i];
			id_len1++;
	    }
	}
	

	char *m_szDevID=(char *)malloc(20);
	__system_property_get("ro.serialno",m_szDevID);
	id_len2=strlen(m_szDevID);
	

	char *temp3=GUID;
	int guid_len=id_len1;
	if(id_len1>id_len2)	{

	    for(int j=0;j<(id_len1-id_len2);j++) {

			*(m_szDevID+j+id_len2)='e'+j;
	    }
	}
	else if(id_len1<id_len2) {

	    for(int k=0;k<(id_len2-id_len1);k++) {

			*(temp2+k+id_len1)='F'+k;
	    }
	    guid_len=id_len2;	
	}

	for(int p=0;p<guid_len;p++)	{

	    if(*(m_szDevID+p)>=*(temp2+p))	
	    	*(temp3+p)=*(temp2+p);
	    else
			*(temp3+p)=*(m_szDevID+p);

	    if((*(temp3+p)<=57)&&(*(temp3+p)>=48)) {

			*(temp3+p)+=30;
	    }
	    else if((*(temp3+p)<=90)&&(*(temp3+p)>=65)) {
			*(temp3+p)=187-(*(temp3+p));
	    }
	    else if((*(temp3+p)<=122)&&(*(temp3+p)>=97)) {
			*(temp3+p)=187-(*(temp3+p));
	    }
	    else {
			*(temp3+p)='E';	
	    }
	}

	//cout <<"MAC :"<<temp2<<"leng:"<<id_len1<<endl;
	//cout <<"SN :"<<m_szDevID<<"leng:"<<id_len2<<endl;
	GUID_LENGTH=guid_len;

	pclose(stream);
	free(m_szDevID);
	m_szDevID=NULL;
	free(temp1);
    temp1 = NULL;
    free(temp2);
    temp2 = NULL;
	return guid_len;
}

int Check_Device_Register_State()
{	
	int Ret=-1;
	if(access(".Key",0)<0)
	    return Ret;

	FILE *file_stream = NULL;
	char *pInfo = NULL;
	file_stream = fopen(".Key", "r+");
	if(NULL == file_stream)
		return Ret;
	
	fseek(file_stream, 0, SEEK_END);
	int fsize = ftell(file_stream);
	pInfo = (char *) malloc(fsize + 1);
	fseek(file_stream, 0, SEEK_SET);
	memset(pInfo,0x00,fsize+1);
	int num_read = fread(pInfo, sizeof(char),fsize,file_stream);
	
	if(num_read!=(GUID_LENGTH*5+1))
	    return Ret;

	int len=GUID_LENGTH;
	char *Buffer1=(char *)malloc(len+1); 
	char *Buffer2=(char *)malloc(len+1); 
	int *cw=(int *)malloc(200);
	char * value=(char *)malloc(10);
	char * temp;
	temp=pInfo;

	for(int i=0;i<len;i++) {   
	    strncpy(value,temp,5);	     
	    temp=temp+5;
	    cw[i]=atoi(value); 		
	}


	rsa_decrypt( Key_n, Key_d, cw, len*5, Buffer1 );
	Get_GUID_For_Device(Buffer2);
	//cout<<"buffer1 is :"<<Buffer1<<endl;
	//cout<<"buffer2 is :"<<Buffer2<<endl;

	if(strncmp(Buffer1,Buffer2,len)==0)
		Ret=0;

	if(NULL != file_stream)
	    fclose(file_stream);

	if(NULL != pInfo) {
	    free(pInfo);
	    pInfo = NULL;
	}

	free(Buffer1);
	free(Buffer2);
	free(value);
	free(cw);
	Buffer1=NULL;
	Buffer2=NULL;
	value=NULL;
	cw=NULL;
	return Ret;
	
}

int Register_For_Device(char * license)
{
	FILE *file_stream = NULL;
        file_stream = fopen(".Key", "w+");
        if(NULL == file_stream)
            return -1;
	//chmod(".Key", S_IRWXO);
	char end='\n';
	fwrite(license, sizeof(char),GUID_LENGTH*5,file_stream);
	fwrite(&end, sizeof(char),1,file_stream);
	if(NULL != file_stream)
		fclose(file_stream);
	return 0;
}

