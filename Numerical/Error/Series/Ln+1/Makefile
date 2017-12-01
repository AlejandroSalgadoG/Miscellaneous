Ln1Main: Ln1.o Ln1Main.o
	g++ $^ -o Ln1Main

Ln1.o: Ln1.C Ln1.h
	g++ -c Ln1.C

Ln1Main.o: Ln1Main.C
	g++ -c Ln1Main.C

clean:
	rm -f *.o Ln1Main

exe:
	./Ln1Main 10 0.1
