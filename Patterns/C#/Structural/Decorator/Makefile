lib=-t:library

mainParam=SuperHero.dll \
		  HiperMan.dll \
		  HeroPower.dll \
		  Fly.dll \
		  XRay.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

flyParam=HeroPower.dll \
		 SuperHero.dll

flyList=$(foreach val,$(flyParam),-r:$(val))

xrayParam=HeroPower.dll \
		 SuperHero.dll

xrayList=$(foreach val,$(xrayParam),-r:$(val))

MainDecorator: MainDecorator.cs $(mainParam)
	gmcs $(mainList) -out:$@ $<

SuperHero.dll: SuperHero.cs
	gmcs $(lib) $<

HiperMan.dll: HiperMan.cs SuperHero.dll
	gmcs $(lib) -r:SuperHero.dll $<

HeroPower.dll: HeroPower.cs SuperHero.dll
	gmcs $(lib) -r:SuperHero.dll $<

Fly.dll: Fly.cs $(flyParam)
	gmcs $(lib) $(flyList) $<

XRay.dll: XRay.cs $(xrayParam)
	gmcs $(lib) $(xrayList) $<

clean:
	rm -f *.dll MainDecorator

exe:
	./MainDecorator
