#pragma once

#include "Jump.h"
#include "Run.h"

class Animal{

    protected:
        Movement * movement;

    public:
        virtual void jump() = 0;
        virtual void run() = 0;

};
