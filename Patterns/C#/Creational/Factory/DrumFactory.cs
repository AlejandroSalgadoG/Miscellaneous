public class DrumFactory : InstrumentFactory{

    public override Instrument createInstrument(){
        return new Drum();
    }

}
