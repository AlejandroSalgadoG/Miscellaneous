Mutex: Mutex.o
	g++ -pthread Mutex.o -o Mutex

Mutex.o: Mutex.C Mutex.h
	g++ -c Mutex.C 

clean:
	rm -f *.o Mutex

exe:
	./Mutex
