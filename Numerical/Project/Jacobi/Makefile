Reader=../../Matrix/Utilities/Reader
Norm=../../Matrix/Utilities/Norm
Dist=../Utilities/Distribute

MainJacobi: Reader.o Norm.o Distribute.o MainJacobi.o
	mpic++ $^ -o MainJacobi

Reader.o: $(Reader).C $(Reader).h
	mpic++ -c $(Reader).C

Norm.o: $(Norm).C $(Norm).h
	mpic++ -c $(Norm).C

Distribute.o: $(Dist).C $(Dist).h
	mpic++ -c $(Dist).C

MainJacobi.o: MainJacobi.C
	mpic++ -I../Utilities -I../../Matrix/Utilities -c MainJacobi.C

clean:
	rm -f *.o MainJacobi

exe:
	mpirun -np 2 ./MainJacobi 4 0.00001 13 matrix.data vector.data
