#include "Fly.h"

void Fly::usePower(){
    cout << "The hero now can fly" << endl;
}

void Fly::setSuperHero(SuperHero * superHero){
    this->superHero = superHero;
}
