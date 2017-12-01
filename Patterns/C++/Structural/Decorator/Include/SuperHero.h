#pragma once

class SuperHero{

    public:
        virtual ~SuperHero();
        virtual void actLikeAHero() = 0;
        virtual void usePower();
        virtual void setSuperHero(SuperHero * superHero);

};
