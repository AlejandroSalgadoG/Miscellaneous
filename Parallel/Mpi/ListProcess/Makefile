CXX=mpic++

ListProcess: ListProcess.o
	$(CXX) ListProcess.o -o ListProcess

ListProcess.o: ListProcess.C
	$(CXX) -c ListProcess.C

clean:
	rm -f *.o ListProcess

exe:
	mpirun -np 8 ./ListProcess
