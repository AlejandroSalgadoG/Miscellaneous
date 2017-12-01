#include "Red.h"

Red::Red(const char* info){
    this->info = info;
}

const char* Red::getInfo(){
    return info;
}

Color* Red::clone(){
    return new Red(info);
}
