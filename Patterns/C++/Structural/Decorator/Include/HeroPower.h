#pragma once

#include "Io.h"
#include "SuperHero.h"

class HeroPower : public SuperHero{

    protected:
        SuperHero * superHero;

    public:
        virtual ~HeroPower();
        void actLikeAHero();
        virtual void usePower() = 0;
        virtual void setSuperHero(SuperHero * superHero) = 0;

};
