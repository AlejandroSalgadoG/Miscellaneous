public class GuitarFactory : InstrumentFactory{

    public override Instrument createInstrument(){
        return new Guitar();
    }

}
