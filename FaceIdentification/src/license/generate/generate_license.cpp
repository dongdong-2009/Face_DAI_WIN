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

#ifdef _WIN32
#include <objbase.h>
#else
#include <uuid/uuid.h>
#endif

using namespace std;


#define MAX_NUM 63001
#define MAX_FILE 1024
#define MAX_PRIME 251
//! 返回代码
#define OK 100
#define ERROR_NOEACHPRIME 101
#define ERROR_NOPUBLICKEY 102
#define ERROR_GENERROR 103

//!　保存私钥d集合
struct pKeyset
{
unsigned int set[MAX_NUM];
unsigned int size;
}pset;

//! 保存公、私钥对
struct pPairkey
{
unsigned int d;
unsigned int e;
unsigned int n;
}pairkey;

int Get_PQD(int& p,int& q,int& d);
bool CheckParse( int argc, char** argv );
bool isPrime( unsigned int m, unsigned int n );
void outputkey();
unsigned int MakePrivatedKeyd( unsigned int uiP, unsigned int uiQ );
unsigned int MakePairkey( unsigned int uiP, unsigned int uiQ, unsigned int uiD );
void rsa_encrypt( int n, int e, char *mw, int mLength, int *&cw );
void rsa_decrypt( int n, int d, int *&cw, int cLength, char *mw );


void outputkey()
{
printf("PublicKey(e,n): (%d,%d)\n",pairkey.e,pairkey.n);
printf("PrivateKey(d,n): (%d,%d)\n",pairkey.d,pairkey.n);
}

// 名称：isPrime
// 功能：判断两个数是否互质
//  参数：m: 数a; n: 数b
// 返回：m、n互质返回true; 否则返回false

bool isPrime( unsigned int m, unsigned int n )
{
unsigned int i=0;
bool Flag = true;

if( m<2 || n<2 )
return false;

unsigned int tem = ( m > n ) ? n : m;
for( i=2; i<=tem && Flag; i++ )
{
bool mFlag = true;
bool nFlag = true;
if( m % i == 0 )
mFlag = false;
if( n % i == 0 )
nFlag = false;
if( !mFlag && !nFlag )
Flag = false;
}
if( Flag )
return true;
else
return false;
}

// 名称：MakePrivatedKeyd
// 功能：由素数Q、Q生成私钥d
//  参数：uiP: 素数P; uiQ: 素数Q
// 返回：私钥d

unsigned int MakePrivatedKeyd( unsigned int uiP, unsigned int uiQ )
{
unsigned int i=0;

//! 得到所有与z互质的数( 私钥d的集合 )
unsigned int z = ( uiP -1 ) * ( uiQ -1 );
pset.size = 0;
for( i=0; i<z; i++ )
{
if( isPrime( i, z ) )
{
pset.set[ pset.size++ ] = i;
}
}

return pset.size;
}

// 名称：MakePairKey
// 功能：生成RSA公、私钥对
//  参数：uiP: 素数P; uiQ: 素数Q; uiD: 私钥d
// 返回：错误代码

unsigned int MakePairkey( unsigned int uiP, unsigned int uiQ, unsigned int uiD )
{
bool bFlag = true;
unsigned int i = 0, e;
unsigned int z = ( uiP-1 ) * ( uiQ-1 );
unsigned int d = pset.set[uiD];
//d=uiD;

if( !isPrime( z, d ) )
return ERROR_NOEACHPRIME;

for( i=2; i<z; i++ )
{
if( (i*d)%z == 1 )
{
e = i;
bFlag = false;
}
}
if( bFlag )
return ERROR_NOPUBLICKEY;

if( (d*e)%z != 1 )
ERROR_GENERROR;

pairkey.d = d;
pairkey.e = e;
pairkey.n = uiP * uiQ;
return OK;
}

int Get_PQD(int& p,int& q,int& d)
{
        FILE *file_stream = NULL;
        char *pInfo = NULL;
        file_stream = fopen("config.ini", "r+");
        if(NULL == file_stream)
        {
            printf("[ERR] File Open Error!\n");
            return -1;
        }
        fseek(file_stream, 0, SEEK_END);
        int fsize = ftell(file_stream);
        pInfo = (char *) malloc(fsize + 1);
        fseek(file_stream, 0, SEEK_SET);
        memset(pInfo,0x00,fsize+1);
        int num_read = fread(pInfo, sizeof(char),fsize,file_stream);
        if(0 == num_read)
        {
            printf("[ERR] File Read Error!\n");
	    return -1;
        }

	int pos=0;
	string str="";

	char* pInfoContent=pInfo;
	while(*pInfoContent!= '\0')
	{
	    // CR+LF
	    if (*pInfoContent == 0x0D)
	    {
		pInfoContent++;
	        if (*pInfoContent == 0x0A)
		{
			if(pos==0)
			{
		    	    p=atoi(str.c_str());
		    	    str="";
		    	    pos=1;	
			}
			else if(pos==1)
			{
			    q=atoi(str.c_str());
			    str="";
			    pos=2;	
			}
			else if(pos==2)
			{
			    d=atoi(str.c_str());
			    str="";
			    break;	
			}
			pInfoContent++;
		}
	    }
	    // LF
	    else if (*pInfoContent == 0x0A)
	    {
		if(pos==0)
		{
	    	    p=atoi(str.c_str());
	    	    str="";
	    	    pos=1;	
		}
		else if(pos==1)
		{
		    q=atoi(str.c_str());
		    str="";
		    pos=2;	
		}
		else if(pos==2)
		{	
		    d=atoi(str.c_str());
		    str="";
		    break;	
		}
		pInfoContent++;
	    }
	    else
	    {
		str += *pInfoContent++;
	    }
	}

	if(NULL != file_stream)
	    fclose(file_stream);   
	if(NULL != pInfo){
	    free(pInfo);
	    pInfo = NULL;
	}
	return 0;
}

// 名称：rsa_encrypt
// 功能：RSA加密运算
//  参数：n: 公钥n; e: 公钥e; mw: 加密明文; iLength: 明文长度; cw: 密文输出
// 返回：无

void rsa_encrypt( int n, int e, char *mw, int mLength, int *&cw )
{
int i=0, j=0;
long temInt = 0;

for( i=0; i<mLength; i++ )
{
temInt = mw[i];
if( e!=0 )
{
for( j=1; j<e; j++ )
{
temInt = ( temInt * mw[i] ) % n;
}
}
else
{
temInt = 1;
}

cw[i] = (int)temInt;
}
}

// 名称：rsa_decrypt
// 功能：RSA解密运算
//  参数：n: 私钥n; d: 私钥d; cw: 密文; cLength: 密文长度; mw: 明文输出
// 返回：无

void rsa_decrypt( int n, int d, int *&cw, int cLength, char *mw )
{
int i=0, j=-1;
long temInt = 0;

for( i=0; i<cLength/4; ++i )
{
mw[i] = 0;
temInt = cw[i];

if( d != 0 )
{
for( j=1; j<d; j++ )
{
temInt = (long)( temInt * cw[i] ) % n;
}
}
else
{
temInt = 1;
}

mw[i] = (char)temInt;
}
}



//! 程序主函数
int main( int argc, char **argv )
{
    int p,q,d;
    Get_PQD(p,q,d);
    cout<<p<<" "<<q<<" "<<d<<endl;
    MakePrivatedKeyd(p, q);
    MakePairkey(p, q, d );
    outputkey();

    int len=25;	
    char *inBuffer=(char *)malloc(len+1); //输入缓冲区
    int *cw=(int *)malloc(MAX_FILE);
    char *temp=inBuffer;
    
    int i;
    if(argc == 1)
    {

    }	
    else if(argc == 2)
    {
	if(strlen(argv[1])<len)
	{
	    strncpy(inBuffer,argv[1],strlen(argv[1]));
	}
	else
	{
	    printf("Parameter error!!!\n");
	    return -1;
	}	
    }
    rsa_encrypt(pairkey.n,pairkey.e, inBuffer, strlen(argv[1]), cw); 

    cout<<"UUID is: ("<<inBuffer<<")"<<endl;

    printf("Liscense is: (");
    for(int j=0;j<strlen(argv[1]);j++)
    {
        printf("%05d",cw[j]);
    }
    printf(")\n");

    free(inBuffer);
    free(cw);
    inBuffer=NULL;
    cw=NULL;	
    return 0;
}
