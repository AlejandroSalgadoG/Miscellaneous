#include "Dog.h"
#include "Cat.h"
#include "Jump.h"
#include "Run.h"

int main(int argc, char *argv[]){

    Animal * dog = new Dog();
    Animal * cat = new Cat();

    dog->jump();
    dog->run();

    cat->jump();
    cat->run();

    delete dog;
    delete cat;

}
