CXX=mpiicpc

Distributed: Distributed.o
	$(CXX) Distributed.o -o Distributed

Distributed.o: Distributed.C
	$(CXX) -c Distributed.C

clean:
	rm -f *.o Distributed

exe:
	mpirun -machinefile machines -np 2 ./Distributed
