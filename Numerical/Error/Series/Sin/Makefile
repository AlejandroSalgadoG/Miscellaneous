SinMain: Factorial.o Sin.o SinMain.o
	g++ $^ -o SinMain

Factorial.o: ../Factorial/Factorial.C ../Factorial/Factorial.h
	cp $^ .
	make $@ -f ../Factorial/Makefile

Sin.o: Sin.C Sin.h
	g++ -c Sin.C

SinMain.o: SinMain.C
	g++ -c SinMain.C

clean:
	rm -f *.o SinMain
	rm -f Factorial*

exe:
	./SinMain 10 1
