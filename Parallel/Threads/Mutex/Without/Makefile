Without: Without.o
	g++ -pthread Without.o -o Without

Without.o: Without.C Without.h
	g++ -c Without.C

clean:
	rm -f *.o Without

exe:
	./Without
