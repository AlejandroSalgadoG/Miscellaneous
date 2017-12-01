#pragma once

#include "Io.h"
#include "Coat.h"

class Jacket : public Coat{

    public:
        const char* getName();
        void coverChest();
};
