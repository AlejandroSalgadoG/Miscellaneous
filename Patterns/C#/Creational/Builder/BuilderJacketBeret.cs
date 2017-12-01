public class BuilderJacketBeret : Builder{

    public BuilderJacketBeret(){
        outfit = new Outfit();
    }
    public override void buildCoat(){
        outfit.addClothe( new Jacket() );
    }

    public override void buildHat(){
        outfit.addClothe( new Beret() );
    }

    public override Outfit getOutfit(){
        return outfit;
    }

}
