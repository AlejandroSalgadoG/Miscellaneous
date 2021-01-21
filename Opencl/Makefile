Main: Main.o
	g++ Main.o -o Main -L/usr/lib/arm-linux-gnueabihf -lOpenCL 

Main.o: Main.cpp
	g++ -c Main.cpp -I /usr/include/

clean:
	rm -f *.o Main

exe:
	sudo ./Main
