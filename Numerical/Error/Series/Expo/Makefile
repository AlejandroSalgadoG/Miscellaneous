ExpoMain: Factorial.o Expo.o ExpoMain.o
	g++ $^ -o ExpoMain

Factorial.o: ../Factorial/Factorial.C ../Factorial/Factorial.h
	cp $^ .
	make $@ -f ../Factorial/Makefile

Expo.o: Expo.C Expo.h
	g++ -c Expo.C

ExpoMain.o: ExpoMain.C
	g++ -c ExpoMain.C

clean:
	rm -f *.o ExpoMain
	rm -f Factorial*

exe:
	./ExpoMain 10 1
