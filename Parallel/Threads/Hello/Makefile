Hello: Hello.o
	g++ -pthread Hello.o -o Hello

Hello.o: Hello.C
	g++ -c Hello.C

clean:
	rm -f *.o Hello

exe:
	./Hello
