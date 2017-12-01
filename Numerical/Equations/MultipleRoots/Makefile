Function=../Functions/Function
Slope=../Functions/SlopeSpecial

MainMultiple: Function.o SlopeSpecial.o Multiple.o MainMultiple.o
	g++ $^ -o MainMultiple

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

SlopeSpecial.o: $(Slope).C $(Slope).h
	g++ -c $(Slope).C

Multiple.o: Multiple.C Multiple.h
	g++ -I../Functions -c Multiple.C

MainMultiple.o: MainMultiple.C
	g++ -c MainMultiple.C

clean:
	rm -f *.o MainMultiple

exe:
	./MainMultiple -3.7 0.0001 40
