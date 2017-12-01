#pragma once

#include "Color.h"

class Red : public Color{

    public:
        Red(const char* info);
        const char* getInfo();
        Color* clone();
};
