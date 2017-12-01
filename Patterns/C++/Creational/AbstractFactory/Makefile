inc=./Include

mainParam=WhiteApartment.o \
		  WhiteHouse.o \
		  FactoryWhite.o \
		  BlackApartment.o \
		  BlackHouse.o \
		  FactoryBlack.o \
		  FactoryCreator.o \
		  MainAbstractFactory.o

factoryParam=FactoryWhite.o \
			 FactoryBlack.o \
			 $(inc)/AbstractFactory.h \
			 $(inc)/FactoryWhite.h \
			 $(inc)/FactoryBlack.h \
			 $(inc)/FactoryCreator.h

whiteParam=WhiteApartment.o \
		   WhiteHouse.o \
		   $(inc)/AbstractFactory.h \
		   $(inc)/Apartment.h \
		   $(inc)/House.h \
		   $(inc)/FactoryWhite.h \

blackParam=BlackApartment.o \
		   BlackHouse.o \
		   $(inc)/AbstractFactory.h \
		   $(inc)/Apartment.h \
		   $(inc)/House.h \
		   $(inc)/FactoryBlack.h \

MainAbstractFactory: $(mainParam)
	g++ $^ -o $@

MainAbstractFactory.o: MainAbstractFactory.C $(inc)/FactoryCreator.h
	g++ -I$(inc) -c $<

FactoryCreator.o: FactoryCreator.C $(factoryParam)
	g++ -I$(inc) -c $<

FactoryWhite.o: FactoryWhite.C $(whiteParam)
	g++ -I$(inc) -c $<

FactoryBlack.o: FactoryBlack.C $(blackParam)
	g++ -I$(inc) -c $<

WhiteApartment.o: WhiteApartment.C $(inc)/WhiteApartment.h $(inc)/Apartment.h
	g++ -I$(inc) -c $<

WhiteHouse.o: WhiteHouse.C $(inc)/WhiteHouse.h $(inc)/Apartment.h
	g++ -I$(inc) -c $<

BlackApartment.o: BlackApartment.C $(inc)/BlackApartment.h $(inc)/House.h
	g++ -I$(inc) -c $<

BlackHouse.o: BlackHouse.C $(inc)/BlackHouse.h $(inc)/House.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainAbstractFactory

exe:
	./MainAbstractFactory
