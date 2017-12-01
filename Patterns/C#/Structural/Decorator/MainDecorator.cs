using System;

public class MainDecorator{

    public static void Main(string[] args){

        SuperHero superHero = new HiperMan();
        superHero.actLikeAHero();

        SuperHero superHeroFly = new Fly();
        SuperHero superHeroXRay = new XRay();

        superHeroFly.setSuperHero(superHero);
        superHeroFly.usePower();

        superHeroXRay.setSuperHero(superHero);
        superHeroXRay.usePower();
    }

}
