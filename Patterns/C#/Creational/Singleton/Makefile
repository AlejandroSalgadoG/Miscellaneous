lib=-t:library

MainSingleton: MainSingleton.cs Id.dll
	gmcs -out:$@ -r:Id.dll $<

Id.dll: Id.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainSingleton

exe:
	./MainSingleton
