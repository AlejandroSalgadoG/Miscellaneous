public abstract class HeroPower : SuperHero{

    protected SuperHero superHero;

    public override void actLikeAHero(){
        superHero.actLikeAHero();
    }

    abstract override public void usePower();
    abstract override public void setSuperHero(SuperHero superHero);

}
