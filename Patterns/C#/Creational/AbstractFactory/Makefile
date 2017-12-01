lib=-t:library

mainParam=FactoryCreator.dll \
		  AbstractFactory.dll \
		  Apartment.dll \
		  House.dll \

mainList=$(foreach val,$(mainParam),-r:$(val))

factoryParam=AbstractFactory.dll \
		     FactoryWhite.dll \
		     FactoryBlack.dll \

factoryList=$(foreach val,$(factoryParam),-r:$(val))

abstractParam=Apartment.dll \
			  House.dll \

abstractList=$(foreach val,$(abstractParam),-r:$(val))

whiteParam=AbstractFactory.dll \
		   Apartment.dll \
	  	   House.dll \
	  	   WhiteApartment.dll \
	  	   WhiteHouse.dll \

whiteList=$(foreach val,$(whiteParam),-r:$(val))

blackParam=AbstractFactory.dll \
		   Apartment.dll \
	  	   House.dll \
	  	   BlackApartment.dll \
	  	   BlackHouse.dll \

blackList=$(foreach val,$(blackParam),-r:$(val))

MainAbstractFactory: MainAbstractFactory.cs $(mainParam)
	gmcs -out:$@ $(mainList) $<

FactoryCreator.dll: FactoryCreator.cs $(factoryParam)
	gmcs $(lib) $(factoryList) $<

AbstractFactory.dll: AbstractFactory.cs $(abstractParam)
	gmcs $(lib) $(abstractList) $<

FactoryWhite.dll: FactoryWhite.cs $(whiteParam)
	gmcs $(lib) $(whiteList) $<

FactoryBlack.dll: FactoryBlack.cs $(blackParam)
	gmcs $(lib) $(blackList) $<

WhiteApartment.dll: WhiteApartment.cs Apartment.dll
	gmcs $(lib) -r:Apartment.dll $<

BlackApartment.dll: BlackApartment.cs Apartment.dll
	gmcs $(lib) -r:Apartment.dll $<

WhiteHouse.dll: WhiteHouse.cs House.dll
	gmcs $(lib) -r:House.dll $<

BlackHouse.dll: BlackHouse.cs House.dll
	gmcs $(lib) -r:House.dll $<

Apartment.dll: Apartment.cs
	gmcs $(lib) $<

House.dll: House.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainAbstractFactory

exe:
	./MainAbstractFactory
