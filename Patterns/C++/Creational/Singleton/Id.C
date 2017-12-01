#include "Id.h"

Id* Id::identity = new Id();

Id::Id(){}

Id* Id::getId(){
    return identity;
}

void Id::consultId(){
    cout << "Your id is 1" << endl;
}
