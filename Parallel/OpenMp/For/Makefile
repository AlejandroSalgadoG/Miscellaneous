CXX=icpc -openmp

For: For.o
	$(CXX) For.o -o For

For.o: For.C
	$(CXX) -c For.C

clean:
	rm -f *.o For

exe:
	./For
