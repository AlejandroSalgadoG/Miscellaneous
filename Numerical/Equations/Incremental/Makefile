Function=../Functions/Function

MainIncremental: Function.o Incremental.o MainIncremental.o
	g++ $^ -o MainIncremental

Function.o: $(Function).C $(Function).h
	g++ -c $(Function).C

Incremental.o: Incremental.C Incremental.h
	g++ -I../Functions -c Incremental.C

MainIncremental.o: MainIncremental.C
	g++ -c MainIncremental.C

clean:
	rm -f *.o MainIncremental

exe:
	./MainIncremental -5 0.1 40
