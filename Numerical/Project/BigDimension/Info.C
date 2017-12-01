#include <sys/sysinfo.h>
#include <fstream>

long int getMem(double rate){
    struct sysinfo info;
    sysinfo(&info);

    return info.freeram * rate;
}

long int getFileSize(char * file){

    std::ifstream file_stream(file, std::ifstream::ate | std::ifstream::binary);

    return file_stream.tellg();
}
