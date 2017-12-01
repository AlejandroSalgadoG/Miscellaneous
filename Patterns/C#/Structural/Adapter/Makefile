lib=-t:library

mainParam=Comunication.dll \
		  Write.dll \
		  Call.dll \
		  CellPhone.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

callParam=Comunication.dll \
		  CellPhone.dll

callList=$(foreach val,$(callParam),-r:$(val))

MainAdapter: MainAdapter.cs $(mainParam)
	gmcs $(mainList) -out:$@ $<

Call.dll: Call.cs $(callParam)
	gmcs $(lib) $(callList) $<

Write.dll: Write.cs Comunication.dll
	gmcs $(lib) -r:Comunication.dll $<

Comunication.dll: Comunication.cs
	gmcs $(lib) $<

CellPhone.dll: CellPhone.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainAdapter

exe:
	./MainAdapter
