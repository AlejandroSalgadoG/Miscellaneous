public class FactoryCreator{

    public static InstrumentFactory createTrumpetFactory(){
        return new TrumpetFactory();
    }

    public static InstrumentFactory createDrumFactory(){
        return new DrumFactory();
    }

    public static InstrumentFactory createGuitarFactory(){
        return new GuitarFactory();
    }

}
