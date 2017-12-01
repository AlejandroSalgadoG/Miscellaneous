public class BuilderVestCap extends Builder{

    BuilderVestCap(){
        outfit = new Outfit();
    }

    public void buildCoat(){
        outfit.addClothe( new Vest() );
    }


    public void buildHat(){
        outfit.addClothe( new Cap() );
    }

    public Outfit getOutfit(){
        return outfit;
    }

}
