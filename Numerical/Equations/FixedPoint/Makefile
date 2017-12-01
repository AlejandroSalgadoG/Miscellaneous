Function=../Functions/Function
Gfunction=../Functions/Gfunction

MainFixed: Function.o Gfunction.o Fixed.o MainFixed.o
	g++ $^ -o MainFixed

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

Gfunction.o: $(Gfunction).C $(Gfunction).h
	g++ -c $(Gfunction).C

Fixed.o: Fixed.C Fixed.h
	g++ -I../Functions -c Fixed.C

MainFixed.o: MainFixed.C
	g++ -c MainFixed.C

clean:
	rm -f *.o MainFixed

exe:
	./MainFixed -3.7 0.00001 40
