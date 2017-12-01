#pragma once

class Worker{

    protected:
        const char* name;

    public:
        Worker(const char* name);
        virtual ~Worker();
        virtual void add(Worker * worker) = 0;
        virtual void remove(Worker * worker) = 0;
        virtual void displayStructure() = 0;
};
