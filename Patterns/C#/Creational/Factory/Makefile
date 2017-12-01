lib=-t:library

mainParam=InstrumentFactory.dll \
		  FactoryCreator.dll \
		  Instrument.dll \

mainList=$(foreach val, $(mainParam), -r:$(val))

creatorParam=InstrumentFactory.dll \
			 TrumpetFactory.dll \
			 DrumFactory.dll \
			 GuitarFactory.dll

creatorList=$(foreach val, $(creatorParam), -r:$(val))

trumpetParam=Instrument.dll \
			 InstrumentFactory.dll \
			 Trumpet.dll

trumpetList=$(foreach val, $(trumpetParam), -r:$(val))

drumParam=Instrument.dll \
		  InstrumentFactory.dll \
		  Drum.dll

drumList=$(foreach val, $(drumParam), -r:$(val))

guitarParam=Instrument.dll \
		    InstrumentFactory.dll \
		    Guitar.dll

guitarList=$(foreach val, $(guitarParam), -r:$(val))

MainFactory: MainFactory.cs $(mainParam)
	gmcs -out:$@ $(mainList) $<

FactoryCreator.dll: FactoryCreator.cs $(creatorParam)
	gmcs $(lib) $(creatorList) $<

TrumpetFactory.dll: TrumpetFactory.cs $(trumpetParam)
	gmcs $(lib) $(trumpetList) $<

DrumFactory.dll: DrumFactory.cs $(drumParam)
	gmcs $(lib) $(drumList) $<

GuitarFactory.dll: GuitarFactory.cs $(guitarParam)
	gmcs $(lib) $(guitarList) $<

InstrumentFactory.dll: InstrumentFactory.cs Instrument.dll
	gmcs $(lib) -r:Instrument.dll $<

Trumpet.dll: Trumpet.cs Instrument.dll
	gmcs $(lib) -r:Instrument.dll $<

Drum.dll: Drum.cs Instrument.dll
	gmcs $(lib) -r:Instrument.dll $<

Guitar.dll: Guitar.cs Instrument.dll
	gmcs $(lib) -r:Instrument.dll $<

Instrument.dll: Instrument.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainFactory

exe:
	./MainFactory
