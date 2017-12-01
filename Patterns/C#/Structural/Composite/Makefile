lib=-t:library

mainParam=Worker.dll \
		  Assistant.dll \
		  Employee.dll \
		  Boss.dll

mainList=$(foreach val,$(mainParam),-r:$(val))

MainComposite: MainComposite.cs $(mainParam)
	gmcs $(mainList) -out:$@ $<

Boss.dll: Boss.cs Worker.dll
	gmcs $(lib) -r:Worker.dll $<

Assistant.dll: Assistant.cs Worker.dll
	gmcs $(lib) -r:Worker.dll $<

Employee.dll: Employee.cs Worker.dll
	gmcs $(lib) -r:Worker.dll $<

Worker.dll: Worker.cs
	gmcs $(lib) $<

clean:
	rm -f *.dll MainComposite

exe:
	./MainComposite
