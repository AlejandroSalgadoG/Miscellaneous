#include "Write.h"
#include "Call.h"

int main(int argc, char *argv[]){

    Comunication * write = new Write();
    Comunication * call = new Call();

    write->comunicate();
    call->comunicate();

    delete write;
    delete call;

}
