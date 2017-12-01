Function=../Functions/Function
Slope=../Functions/SlopeSpecial

MainSecant: Function.o SlopeSpecial.o Secant.o MainSecant.o
	g++ $^ -o MainSecant

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

SlopeSpecial.o: $(Slope).C $(Slope).h
	g++ -c $(Slope).C

Secant.o: Secant.C Secant.h
	g++ -I../Functions -c Secant.C

MainSecant.o: MainSecant.C
	g++ -c MainSecant.C

clean:
	rm -f *.o MainSecant

exe:
	./MainSecant -3.7 -3.6 0.00001 40 rel
