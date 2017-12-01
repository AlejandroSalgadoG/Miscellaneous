inc=./Include

mainParam=Trumpet.o \
		  Drum.o \
		  Guitar.o \
		  TrumpetFactory.o \
		  DrumFactory.o \
		  GuitarFactory.o \
		  FactoryCreator.o \
		  MainFactory.o

creatorParam=TrumpetFactory.o \
			 DrumFactory.o \
			 GuitarFactory.o \
			 $(inc)/FactoryCreator.h

trumpetParam=Trumpet.o \
			 $(inc)/TrumpetFactory.h \
			 $(inc)/InstrumentFactory.h

drumParam=Drum.o \
		  $(inc)/DrumFactory.h \
		  $(inc)/InstrumentFactory.h

guitarParam=Guitar.o \
			$(inc)/GuitarFactory.h \
			$(inc)/InstrumentFactory.h

MainFactory: $(mainParam)
	g++ $^ -o $@

MainFactory.o: MainFactory.C FactoryCreator.o
	g++ -I$(inc) -c $<

FactoryCreator.o: FactoryCreator.C $(creatorParam)
	g++ -I$(inc) -c $<

TrumpetFactory.o: TrumpetFactory.C $(trumpetParam)
	g++ -I$(inc) -c $<

DrumFactory.o: DrumFactory.C $(drumParam)
	g++ -I$(inc) -c $<

GuitarFactory.o: GuitarFactory.C $(guitarParam)
	g++ -I$(inc) -c $<

Trumpet.o: Trumpet.C $(inc)/Trumpet.h $(inc)/Instrument.h
	g++ -I$(inc) -c $<

Drum.o: Drum.C $(inc)/Drum.h $(inc)/Instrument.h
	g++ -I$(inc) -c $<

Guitar.o: Guitar.C $(inc)/Guitar.h $(inc)/Instrument.h
	g++ -I$(inc) -c $<


clean:
	rm -f *.o MainFactory

exe:
	./MainFactory
