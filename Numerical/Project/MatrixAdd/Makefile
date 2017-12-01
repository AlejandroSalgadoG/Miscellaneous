Dist=../Utilities/Distribute
Reader=../../Matrix/Utilities/Reader

MatrixAdd: Initializer.o Reader.o Distribute.o MatrixAdd.o
	mpic++ $^ -o MatrixAdd

Initializer.o: Initializer.C Initializer.h
	mpic++ -c Initializer.C

Reader.o: $(Reader).C $(Reader).h
	mpic++ -c $(Reader).C

Distribute.o: $(Dist).C $(Dist).h
	mpic++ -c $(Dist).C

MatrixAdd.o: MatrixAdd.C
	mpic++ -I../Utilities -I../../Matrix/Utilities -c MatrixAdd.C

clean:
	rm -f *.o MatrixAdd

exe:
	mpirun -np 3 ./MatrixAdd
