param=Instrument.class \
	  DrumFactory.java \
	  Guitar.java \
	  Trumpet.java \
	  Drum.java \
	  GuitarFactory.java \
	  InstrumentFactory.java \
	  TrumpetFactory.java \
	  FactoryCreator.java

MainFactory.class: $(param) MainFactory.java
	javac MainFactory.java

Tumpet.class: Trumpet.java Instrument.class
	javac $<

Instrument.class: Instrument.java
	javac $<

clean:
	rm -f *.class

exe:
	java MainFactory
