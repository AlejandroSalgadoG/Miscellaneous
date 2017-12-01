#include "BuilderJacketCap.h"

BuilderJacketCap::BuilderJacketCap(){
    outfit = new Outfit();
}

void BuilderJacketCap::buildCoat(){
    outfit->addClothe( new Jacket() );
}

void BuilderJacketCap::buildHat(){
    outfit->addClothe( new Cap() );
}

Outfit* BuilderJacketCap::getOutfit(){
    return outfit;
}
