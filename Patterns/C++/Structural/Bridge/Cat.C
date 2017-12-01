#include "Cat.h"

void Cat::jump(){

    movement = new Jump();

    cout << "The cat ";
    movement->move();

    delete movement;
}

void Cat::run(){

    movement = new Run();


    cout << "The cat ";
    movement->move();

    delete movement;
}
