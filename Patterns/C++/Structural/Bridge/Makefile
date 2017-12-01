inc=./Include

mainParam=Dog.o \
		  Cat.o \
		  Jump.o \
		  Run.o \
		  MainBridge.o

mainBridgeParam=$(inc)/Animal.h \
				$(inc)/Dog.h \
				Dog.o \
				$(inc)/Cat.h \
				Cat.o \
				$(inc)/Jump.h \
				Jump.o \
				$(inc)/Run.h \
				Run.o

MainBridge: $(mainParam)
	g++ $^ -o $@

MainBridge.o: MainBridge.C $(mainBridgeParam)
	g++ -I$(inc) -c $<

Dog.o: Dog.C $(inc)/Dog.h $(inc)/Animal.h
	g++ -I$(inc) -c $<

Cat.o: Cat.C $(inc)/Cat.h $(inc)/Animal.h
	g++ -I$(inc) -c $<

Jump.o: Jump.C $(inc)/Jump.h $(inc)/Movement.h
	g++ -I$(inc) -c $<

Run.o: Run.C $(inc)/Run.h $(inc)/Movement.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainBridge

exe:
	./MainBridge
