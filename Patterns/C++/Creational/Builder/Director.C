#include "Director.h"

Builder* Director::direct(Builder* builder){

    builder->buildCoat();
    builder->buildHat();

    return builder;
}
