#pragma once

#include "Builder.h"
#include "Outfit.h"
#include "Jacket.h"
#include "Beret.h"

class BuilderJacketBeret : public Builder{

    public:
        BuilderJacketBeret();
        void buildCoat();
        void buildHat();
        Outfit * getOutfit();

};
