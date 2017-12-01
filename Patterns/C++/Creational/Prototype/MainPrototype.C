#include "Background.h"
#include "Yellow.h"
#include "Blue.h"
#include "Red.h"

int main(int argc, char *argv[]){

    Background * background = new Background();

    Color * yellow = new Yellow("color yellow");
    Color * blue = new Blue("color blue");
    Color * red = new Red("color red");

    background->setBackground(yellow);
    background->setBackground(blue);
    background->setBackground(red);

    delete background;
    delete yellow;
    delete blue;
    delete red;
}
