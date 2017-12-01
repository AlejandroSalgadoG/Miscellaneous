MainCuadratic: Cuadratic.o MainCuadratic.o
	mpic++ $^ -o MainCuadratic

MainCuadratic.o: MainCuadratic.C Types.h
	mpic++ -c MainCuadratic.C

Cuadratic.o: Cuadratic.C Cuadratic.h Types.h
	mpic++ -c Cuadratic.C

clean:
	rm -f *.o MainCuadratic

exe:
	mpirun -np 2 ./MainCuadratic 1 0 -1
