LnMain: Ln.o LnMain.o
	g++ $^ -o LnMain

Ln.o: Ln.C Ln.h
	g++ -c Ln.C

LnMain.o: LnMain.C
	g++ -c LnMain.C

clean:
	rm -f *.o LnMain

exe:
	./LnMain 10 1.1
