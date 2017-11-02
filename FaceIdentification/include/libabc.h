#ifndef LIBABC_H
#define LIBABC_H

int Check_Device_Register_State(char* path);
int Get_GUID_For_Device(char * GUID);
int Register_For_Device(char* path,char * license);

#endif //LIBABC_H