Tan_1Main: Tan_1.o Tan_1Main.o
	g++ $^ -o Tan_1Main

Tan_1.o: Tan_1.C Tan_1.h
	g++ -c Tan_1.C

Tan_1Main.o: Tan_1Main.C
	g++ -c Tan_1Main.C

clean:
	rm -f *.o Tan_1Main

exe:
	./Tan_1Main 10 1
