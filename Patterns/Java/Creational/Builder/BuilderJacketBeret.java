public class BuilderJacketBeret extends Builder{

    BuilderJacketBeret(){
        outfit = new Outfit();
    }

    public void buildCoat(){
       outfit.addClothe( new Jacket() );
    }


    public void buildHat(){
        outfit.addClothe( new Beret() );
    }

    public Outfit getOutfit(){
        return outfit;
    }

}
