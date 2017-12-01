lib=-t:library

mainParam=Animal.dll \
		  Dog.dll \
		  Cat.dll \
		  Movement.dll \
		  Jump.dll \
		  Run.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

dogParam=Animal.dll \
		 Jump.dll \
		 Run.dll \
		 Movement.dll

dogList=$(foreach val,$(dogParam),-r:$(val))

catParam=Animal.dll \
		 Jump.dll \
		 Run.dll \
		 Movement.dll

catList=$(foreach val,$(catParam),-r:$(val))

MainBridge: MainBridge.cs $(mainParam)
	gmcs $(mainList) -out:MainBridge $<

Dog.dll: Dog.cs $(dogParam)
	gmcs $(lib) $(dogList) $<

Cat.dll: Cat.cs $(catParam)
	gmcs $(lib) $(catList) $<

Animal.dll: Animal.cs Movement.dll
	gmcs $(lib) -r:Movement.dll $<

Jump.dll: Jump.cs Movement.dll
	gmcs $(lib) -r:Movement.dll $<

Run.dll: Run.cs Movement.dll
	gmcs $(lib) -r:Movement.dll $<

Movement.dll: Movement.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainBridge

exe:
	./MainBridge
