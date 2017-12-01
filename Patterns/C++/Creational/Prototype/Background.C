#include "Background.h"

void Background::setBackground(Color * color){
    Color * colorCloned = color->clone();
    cout << "The background is " << colorCloned->getInfo() << endl;
    delete colorCloned;
}
