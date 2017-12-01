public class TrumpetFactory : InstrumentFactory{

    public override Instrument createInstrument(){
        return new Trumpet();
    }

}
