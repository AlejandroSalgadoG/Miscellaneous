public class MainBuilder{

    public static void main(String[] args){

        Builder builderJacketBeret = Director.direct( new BuilderJacketBeret() );
        Builder builderJacketCap = Director.direct( new BuilderJacketCap() );
        Builder builderVestBeret = Director.direct( new BuilderVestBeret() );
        Builder builderVestCap = Director.direct( new BuilderVestCap() );

        Outfit jacketBeret = builderJacketBeret.getOutfit();
        Outfit jacketCap = builderJacketCap.getOutfit();
        Outfit vestBeret = builderVestBeret.getOutfit();
        Outfit vestCap = builderVestCap.getOutfit();

        jacketBeret.listClothe();
        jacketCap.listClothe();
        vestBeret.listClothe();
        vestCap.listClothe();

    }

}
