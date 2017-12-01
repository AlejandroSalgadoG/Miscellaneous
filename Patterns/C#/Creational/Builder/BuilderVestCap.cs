public class BuilderVestCap : Builder{

    public BuilderVestCap(){
        outfit = new Outfit();
    }

    public override void buildCoat(){
        outfit.addClothe( new Vest() );
    }

    public override void buildHat(){
        outfit.addClothe( new Cap() );
    }

    public override Outfit getOutfit(){
        return outfit;
    }

}
