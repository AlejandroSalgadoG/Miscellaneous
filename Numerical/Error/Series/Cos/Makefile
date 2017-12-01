CosMain: Factorial.o Cos.o CosMain.o
	g++ $^ -o CosMain

Factorial.o: ../Factorial/Factorial.C ../Factorial/Factorial.h
	cp $^ .
	make $@ -f ../Factorial/Makefile

Cos.o: Cos.C Cos.h
	g++ -c Cos.C

CosMain.o: CosMain.C
	g++ -c CosMain.C

clean:
	rm -f *.o CosMain
	rm -f Factorial*

exe:
	./CosMain 10 0.5
