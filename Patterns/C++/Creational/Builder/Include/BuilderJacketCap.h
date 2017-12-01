#pragma once

#include "Builder.h"
#include "Outfit.h"
#include "Jacket.h"
#include "Cap.h"

class BuilderJacketCap : public Builder{

    public:
        BuilderJacketCap();
        void buildCoat();
        void buildHat();
        Outfit * getOutfit();

};
