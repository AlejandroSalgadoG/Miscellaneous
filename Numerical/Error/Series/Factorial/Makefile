FactorialMain: Factorial.o FactorialMain.o
	g++ $^ -o FactorialMain

Factorial.o: Factorial.C Factorial.h
	g++ -c Factorial.C

FactorialMain.o: FactorialMain.C
	g++ -c FactorialMain.C

clean:
	rm -f *.o FactorialMain

exe:
	./FactorialMain 5
