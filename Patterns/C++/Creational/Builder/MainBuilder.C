#include "Director.h"
#include "BuilderJacketBeret.h"
#include "BuilderJacketCap.h"
#include "BuilderVestBeret.h"
#include "BuilderVestCap.h"

int main(int argc, char *argv[]){

   Builder * builderJacketBeret = Director::direct( new BuilderJacketBeret() );
   Builder * builderJacketCap = Director::direct( new BuilderJacketCap() );
   Builder * builderVestBeret = Director::direct( new BuilderVestBeret() );
   Builder * builderVestCap = Director::direct( new BuilderVestCap() );

   Outfit * jacketBeret = builderJacketBeret->getOutfit();
   Outfit * jacketCap = builderJacketCap->getOutfit();
   Outfit * vestBeret = builderVestBeret->getOutfit();
   Outfit * vestCap = builderVestCap->getOutfit();

   jacketBeret->listClothe();
   jacketCap->listClothe();
   vestBeret->listClothe();
   vestCap->listClothe();

   delete builderJacketBeret;
   delete builderJacketCap;
   delete builderVestBeret;
   delete builderVestCap;

   delete jacketBeret;
   delete jacketCap;
   delete vestBeret;
   delete vestCap;

}
