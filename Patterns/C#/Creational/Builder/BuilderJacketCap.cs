public class BuilderJacketCap : Builder{

    public BuilderJacketCap(){
        outfit = new Outfit();
    }

    public override void buildCoat(){
        outfit.addClothe( new Jacket() );
    }

    public override void buildHat(){
        outfit.addClothe( new Cap() );
    }

    public override Outfit getOutfit(){
        return outfit;
    }

}
