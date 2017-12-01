using System;

public class Fly : HeroPower{

    public override void usePower(){
        Console.WriteLine("The super hero now can fly");
    }

    public override void setSuperHero(SuperHero superHero){
        this.superHero = superHero;
    }

}
