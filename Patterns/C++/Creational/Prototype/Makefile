inc=./Include

mainParam=Color.o \
		  Background.o \
		  Yellow.o \
		  Blue.o \
		  Red.o \
		  MainPrototype.o

protoParam=Background.o \
		   Yellow.o \
		   Blue.o \
		   Red.o

MainPrototype: $(mainParam)
	g++ $^ -o $@

MainPrototype.o: MainPrototype.C $(protoParam)
	g++ -I$(inc) -c $<

Background.o: Background.C $(inc)/Background.h Color.o
	g++ -I$(inc) -c $<

Yellow.o: Yellow.C $(inc)/Yellow.h Color.o
	g++ -I$(inc) -c $<

Blue.o: Blue.C $(inc)/Blue.h Color.o
	g++ -I$(inc) -c $<

Red.o: Red.C $(inc)/Red.h Color.o
	g++ -I$(inc) -c $<

Color.o: Color.C $(inc)/Color.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainPrototype

exe:
	./MainPrototype
