using System;

public class XRay : HeroPower{

    public override void usePower(){
        Console.WriteLine("The super hero now can use x-ray");
    }

    public override void setSuperHero(SuperHero superHero){
        this.superHero = superHero;
    }

}
