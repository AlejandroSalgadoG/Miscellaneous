#pragma once

#include "Color.h"

class Yellow : public Color{

    public:
        Yellow(const char* info);
        const char* getInfo();
        Color* clone();
};
