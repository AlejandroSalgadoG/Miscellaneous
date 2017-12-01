#pragma once

#include "Comunication.h"
#include "CellPhone.h"

class Call : public Comunication{

    CellPhone * cellPhone;

    public:
        Call();
        ~Call();
        void comunicate();

};
