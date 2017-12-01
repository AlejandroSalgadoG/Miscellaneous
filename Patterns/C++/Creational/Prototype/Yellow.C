#include "Yellow.h"

Yellow::Yellow(const char* info){
    this->info = info;
}

const char* Yellow::getInfo(){
    return info;
}

Color* Yellow::clone(){
    return new Yellow(info);
}
