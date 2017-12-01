Function=../Functions/Function

MainNewton: Function.o Newton.o MainNewton.o
	g++ $^ -o MainNewton

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

Newton.o: Newton.C Newton.h
	g++ -I../Functions -c Newton.C

MainNewton.o: MainNewton.C
	g++ -c MainNewton.C

clean:
	rm -f *.o MainNewton

exe:
	./MainNewton -3.7 0.0001 40
