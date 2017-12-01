Function=../Functions/Function
Slope=../Functions/SlopeSpecial

MainFalse: Function.o SlopeSpecial.o FalsePos.o MainFalsePos.o
	g++ $^ -o MainFalsePos

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

SlopeSpecial.o: $(Slope).C $(Slope).h
	g++ -c $(Slope).C

FalsePos.o: FalsePos.C FalsePos.h
	g++ -I../Functions -c FalsePos.C

MainFalsePos.o: MainFalsePos.C
	g++ -c MainFalsePos.C

clean:
	rm -f *.o MainFalsePos

exe:
	./MainFalsePos -3.7 -3.6 0.00001 40
