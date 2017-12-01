public class BuilderVestBeret extends Builder{

    BuilderVestBeret(){
        outfit = new Outfit();
    }

    public void buildCoat(){
        outfit.addClothe( new Vest() );
    }


    public void buildHat(){
        outfit.addClothe( new Beret() );
    }

    public Outfit getOutfit(){
        return outfit;
    }

}
