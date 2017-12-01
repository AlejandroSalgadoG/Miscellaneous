#pragma once

#include "Color.h"

class Blue : public Color{

    public:
        Blue(const char* info);
        const char* getInfo();
        Color* clone();
};
