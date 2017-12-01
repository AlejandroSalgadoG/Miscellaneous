Float: Float.o
	g++ Float.o -o Float

Float.o: Float.C
	g++ -c -g Float.C

clean:
	rm -f *.o Float

exe:
	./Float

gdb:
	make
	gdb ./Float
