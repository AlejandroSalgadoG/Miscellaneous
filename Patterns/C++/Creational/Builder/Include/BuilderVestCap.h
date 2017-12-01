#pragma once

#include "Builder.h"
#include "Outfit.h"
#include "Vest.h"
#include "Cap.h"

class BuilderVestCap : public Builder{

    public:
        BuilderVestCap();
        void buildCoat();
        void buildHat();
        Outfit * getOutfit();

};
