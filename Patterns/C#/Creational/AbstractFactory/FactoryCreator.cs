public class FactoryCreator{

    public static AbstractFactory createFactoryWhite(){
        return new FactoryWhite();
    }

    public static AbstractFactory createFactoryBlack(){
        return new FactoryBlack();
    }

}
