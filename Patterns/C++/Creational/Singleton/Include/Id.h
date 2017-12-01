#pragma once

#include "Io.h"

class Id{

    static Id* identity;

    Id();

    public:
        static Id* getId();
        void consultId();

};
