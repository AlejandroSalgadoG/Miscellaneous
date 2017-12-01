inc=./Include

mainParam=SuperHero.o \
		  HiperMan.o \
		  HeroPower.o \
		  Fly.o \
		  XRay.o \
		  MainDecorator.o

mainDecorator=HeroPower.o \
     		  Fly.o \
     		  XRay.o

MainDecorator: $(mainParam)
	g++ $^ -o $@

MainDecorator.o: MainDecorator.C $(mainDecorator)
	g++ -I$(inc) -c $<

SuperHero.o: SuperHero.C $(inc)/SuperHero.h
	g++ -I$(inc) -c $<

HeroPower.o: HeroPower.C $(inc)/HeroPower.h
	g++ -I$(inc) -c $<

HiperMan.o: HiperMan.C $(inc)/HiperMan.h
	g++ -I$(inc) -c $<

Fly.o: Fly.C $(inc)/Fly.h
	g++ -I$(inc) -c $<

XRay.o: XRay.C $(inc)/XRay.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainDecorator

exe:
	./MainDecorator
