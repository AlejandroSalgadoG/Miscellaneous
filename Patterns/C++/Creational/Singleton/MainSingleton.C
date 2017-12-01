#include "Id.h"

int main(int argc, char *argv[]){

    Id * id = Id::getId();
    id->consultId();
    delete id;

}
