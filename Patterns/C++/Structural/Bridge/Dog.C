#include "Dog.h"

void Dog::jump(){

    movement = new Jump();

    cout << "The dog ";
    movement->move();

    delete movement;
}

void Dog::run(){

    movement = new Run();


    cout << "The dog ";
    movement->move();

    delete movement;
}
