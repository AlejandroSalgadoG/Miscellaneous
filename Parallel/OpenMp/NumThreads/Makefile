CXX=icpc -openmp

NumThreads: NumThreads.o
	$(CXX) NumThreads.o -o NumThreads

NumThreads.o: NumThreads.C
	$(CXX) -c NumThreads.C

clean:
	rm -f *.o NumThreads

exe:
	./NumThreads
