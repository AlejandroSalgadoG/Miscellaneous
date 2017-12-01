#include "HiperMan.h"
#include "Fly.h"
#include "XRay.h"

int main(int argc, char *argv[]){

    SuperHero * superHero = new HiperMan();
    superHero->actLikeAHero();

    SuperHero * superHeroFly = new Fly();
    SuperHero * superHeroXRay = new XRay();

    superHeroFly->setSuperHero(superHero);
    superHeroFly->usePower();

    superHeroXRay->setSuperHero(superHero);
    superHeroXRay->usePower();

	delete superHero;
	delete superHeroFly;
	delete superHeroXRay;

}
