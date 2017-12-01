#pragma once

class Color{

    protected:
        const char* info;

    public:
        virtual ~Color();
        virtual const char* getInfo() = 0;
        virtual Color* clone() = 0;
};
