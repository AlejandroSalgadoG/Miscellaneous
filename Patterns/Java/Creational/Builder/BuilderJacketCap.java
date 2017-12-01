public class BuilderJacketCap extends Builder{

    BuilderJacketCap(){
        outfit = new Outfit();
    }

    public void buildCoat(){
        outfit.addClothe( new Jacket() );
    }


    public void buildHat(){
        outfit.addClothe( new Cap() );
    }

    public Outfit getOutfit(){
        return outfit;
    }

}
