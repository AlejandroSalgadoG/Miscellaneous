lib=-t:library

mainParam=Builder.dll \
		  Outfit.dll \
		  BuilderJacketBeret.dll \
		  BuilderJacketCap.dll \
		  BuilderVestBeret.dll \
		  BuilderVestCap.dll \
		  Director.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

jacketBeretParam=Builder.dll \
				 Outfit.dll \
				 Clothe.dll \
				 Coat.dll \
				 Hat.dll \
				 Jacket.dll \
				 Beret.dll

jacketBeretList=$(foreach val,$(jacketBeretParam), -r:$(val))

jacketCapParam=Builder.dll \
			   Outfit.dll \
			   Clothe.dll \
			   Coat.dll \
			   Hat.dll \
			   Jacket.dll \
			   Cap.dll

jacketCapList=$(foreach val,$(jacketCapParam), -r:$(val))

vestBeretParam=Builder.dll \
			   Outfit.dll \
			   Clothe.dll \
			   Coat.dll \
			   Hat.dll \
			   Vest.dll \
			   Beret.dll

vestBeretList=$(foreach val,$(vestBeretParam), -r:$(val))

vestCapParam=Builder.dll \
			 Outfit.dll \
			 Clothe.dll \
			 Coat.dll \
			 Hat.dll \
			 Vest.dll \
			 Cap.dll

vestCapList=$(foreach val,$(vestCapParam), -r:$(val))

MainBuilder: MainBuilder.cs $(mainParam)
	gmcs -out:$@ $(mainList) $<

BuilderJacketBeret.dll: BuilderJacketBeret.cs $(jacketBeretParam)
	gmcs $(lib) $(jacketBeretList) $<

BuilderJacketCap.dll: BuilderJacketCap.cs $(jacketCapParam)
	gmcs $(lib) $(jacketCapList) $<

BuilderVestBeret.dll: BuilderVestBeret.cs $(vestBeretParam)
	gmcs $(lib) $(vestBeretList) $<

BuilderVestCap.dll: BuilderVestCap.cs $(vestCapParam)
	gmcs $(lib) $(vestCapList) $<

Director.dll: Director.cs Builder.dll
	gmcs $(lib) -r:Builder.dll $<

Builder.dll: Builder.cs Outfit.dll
	gmcs $(lib) -r:Outfit.dll $<

Outfit.dll: Outfit.cs Clothe.dll
	gmcs $(lib) -r:Clothe.dll $<

Jacket.dll: Jacket.cs Clothe.dll Coat.dll
	gmcs $(lib) -r:Clothe.dll,Coat.dll $<

Vest.dll: Vest.cs Clothe.dll Coat.dll
	gmcs $(lib) -r:Clothe.dll,Coat.dll $<

Cap.dll: Cap.cs Clothe.dll Hat.dll
	gmcs $(lib) -r:Clothe.dll,Hat.dll $<

Beret.dll: Beret.cs Clothe.dll Hat.dll
	gmcs $(lib) -r:Clothe.dll,Hat.dll $<

Coat.dll: Coat.cs Clothe.dll
	gmcs $(lib) -r:Clothe.dll $<

Hat.dll: Hat.cs Clothe.dll
	gmcs $(lib) -r:Clothe.dll $<

Clothe.dll: Clothe.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainBuilder

make exe:
	./MainBuilder
