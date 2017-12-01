CXX=mpic++

Initialization: Initialization.o
	$(CXX) Initialization.o -o Initialization

Initialization.o: Initialization.C
	$(CXX) -c Initialization.C

clean:
	rm -f *.o Initialization

exe:
	mpirun -np 2 ./Initialization
