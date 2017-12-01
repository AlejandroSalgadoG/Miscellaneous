public class BuilderVestBeret : Builder{

    public BuilderVestBeret(){
        outfit = new Outfit();
    }

    public override void buildCoat(){
        outfit.addClothe( new Vest() );
    }

    public override void buildHat(){
        outfit.addClothe( new Beret() );
    }

    public override Outfit getOutfit(){
        return outfit;
    }

}
