#include "Blue.h"

Blue::Blue(const char* info){
    this->info = info;
}

const char* Blue::getInfo(){
    return info;
}

Color* Blue::clone(){
    return new Blue(info);
}
