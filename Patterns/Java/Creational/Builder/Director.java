public class Director{

    public static Builder direct(Builder builder){

        builder.buildCoat();
        builder.buildHat();

        return builder;
    }

}
