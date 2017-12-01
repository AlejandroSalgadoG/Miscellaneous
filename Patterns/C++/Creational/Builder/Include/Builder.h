#pragma once

#include "Outfit.h"

class Builder{

    protected:
        Outfit * outfit;

    public:
        virtual void buildCoat() = 0;
        virtual void buildHat() = 0;
        virtual Outfit * getOutfit() = 0;

};
