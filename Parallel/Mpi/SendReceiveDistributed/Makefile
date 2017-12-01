CXX=mpic++

SendReceiveDistributed: SendReceiveDistributed.o
	$(CXX) SendReceiveDistributed.o -o SendReceiveDistributed

SendReceiveDistributed.o: SendReceiveDistributed.C
	$(CXX) -c SendReceiveDistributed.C

clean:
	rm -f *.o SendReceiveDistributed

exe:
	mpirun -machinefile machines -np 2 ./SendReceiveDistributed
