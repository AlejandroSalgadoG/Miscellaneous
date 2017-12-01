#include "BuilderJacketBeret.h"

BuilderJacketBeret::BuilderJacketBeret(){
    outfit = new Outfit();
}

void BuilderJacketBeret::buildCoat(){
    outfit->addClothe( new Jacket() );
}

void BuilderJacketBeret::buildHat(){
    outfit->addClothe( new Beret() );
}

Outfit* BuilderJacketBeret::getOutfit(){
    return outfit;
}
