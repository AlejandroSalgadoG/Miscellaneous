lib=-t:library

mainParam=Color.dll \
		  Background.dll \
		  Yellow.dll \
		  Blue.dll \
		  Red.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

MainPrototype: MainPrototype.cs $(mainParam)
	gmcs -out:$@ $(mainList) $<

Background.dll: Background.cs Color.dll
	gmcs $(lib) -r:Color.dll $<

Yellow.dll: Yellow.cs Color.dll
	gmcs $(lib) -r:Color.dll $<

Blue.dll: Blue.cs Color.dll
	gmcs $(lib) -r:Color.dll $<

Red.dll: Red.cs Color.dll
	gmcs $(lib) -r:Color.dll $<

Color.dll: Color.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainPrototype

exe:
	./MainPrototype
