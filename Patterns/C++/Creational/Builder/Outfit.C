#include "Outfit.h"

Outfit::~Outfit(){
    for(Clothe * clothe: shoped)
        delete clothe;
}

void Outfit::addClothe(Clothe * clothe){

    shoped.push_back(clothe);

}

void Outfit::listClothe(){
    cout << "Outfit:" << endl;
    for(Clothe * clothe: shoped)
        cout << "\t " << clothe->getName() << endl;
}
