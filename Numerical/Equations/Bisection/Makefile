Function=../Functions/Function

MainBisection: Function.o Bisection.o MainBisection.o
	g++ $^ -o MainBisection

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

Bisection.o: Bisection.C Bisection.h
	g++ -I../Functions -c Bisection.C

MainBisection.o: MainBisection.C
	g++ -c MainBisection.C

clean:
	rm -f *.o MainBisection

exe:
	./MainBisection -3.7 -3.6 0.00001 40
