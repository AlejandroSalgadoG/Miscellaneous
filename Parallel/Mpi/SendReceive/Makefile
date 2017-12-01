CXX=mpic++

SendReceive: SendReceive.o
	$(CXX) SendReceive.o -o SendReceive

SendReceive.o: SendReceive.C
	$(CXX) -c SendReceive.C

clean:
	rm -f *.o SendReceive

exe:
	mpirun -np 2 ./SendReceive
