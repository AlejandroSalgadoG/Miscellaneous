#pragma once

#include "Builder.h"
#include "Outfit.h"
#include "Vest.h"
#include "Beret.h"

class BuilderVestBeret : public Builder{

    public:
        BuilderVestBeret();
        void buildCoat();
        void buildHat();
        Outfit * getOutfit();

};
