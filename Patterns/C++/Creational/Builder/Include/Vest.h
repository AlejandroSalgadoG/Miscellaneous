#pragma once

#include "Io.h"
#include "Coat.h"

class Vest : public Coat{

    public:
        const char* getName();
        void coverChest();

};
