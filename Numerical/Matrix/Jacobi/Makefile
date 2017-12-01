Reader=../Utilities/Reader
Norm=../Utilities/Norm

MainJacobi: Reader.o Norm.o Jacobi.o MainJacobi.o
	g++ $^ -o MainJacobi

Reader.o: $(Reader).C $(Reader).h
	g++ -c $(Reader).C

Norm.o: $(Norm).C $(Norm).h
	g++ -c $(Norm).C

Jacobi.o: Jacobi.C Jacobi.h
	g++ -I../Utilities -c Jacobi.C

MainJacobi.o: MainJacobi.C
	g++ -I../Utilities -c MainJacobi.C

clean:
	rm -f *.o MainJacobi

exe:
	./MainJacobi 4 0.00001 13 matrix.data vector.data
