public class MainFactory{

    public static void Main(string[] args){

        InstrumentFactory trumpetFactory = FactoryCreator.createTrumpetFactory();
        InstrumentFactory drumFactory = FactoryCreator.createDrumFactory();
        InstrumentFactory guitarFactory = FactoryCreator.createGuitarFactory();

        Instrument trumpet = trumpetFactory.createInstrument();
        Instrument drum = drumFactory.createInstrument();
        Instrument guitar = guitarFactory.createInstrument();

        trumpet.play();
        drum.play();
        guitar.play();

    }

}
