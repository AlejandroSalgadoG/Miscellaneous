inc=./Include

mainParam=Id.o \
		  MainSingleton.o

MainSingleton: $(mainParam)
	g++ $^ -o $@

MainSingleton.o: MainSingleton.C Id.o
	g++ -I$(inc) -c $<

Id.o: Id.C $(inc)/Id.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainSingleton

exe:
	./MainSingleton
