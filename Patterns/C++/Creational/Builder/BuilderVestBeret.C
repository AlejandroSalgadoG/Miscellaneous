#include "BuilderVestBeret.h"

BuilderVestBeret::BuilderVestBeret(){
    outfit = new Outfit();
}

void BuilderVestBeret::buildCoat(){
    outfit->addClothe( new Vest() );
}

void BuilderVestBeret::buildHat(){
    outfit->addClothe( new Beret() );
}

Outfit* BuilderVestBeret::getOutfit(){
    return outfit;
}
