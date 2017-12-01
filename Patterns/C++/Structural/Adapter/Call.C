#include "Call.h"

Call::Call(){
    cellPhone = new CellPhone();
}

Call::~Call(){
    delete cellPhone;
}

void Call::comunicate(){
    cellPhone->makePhoneCall();
}
