Reader=../../Matrix/Utilities/Reader
VectorGen=../Utilities/VectorGen
MatrixGen=../Utilities/MatrixGen

all: MainVectorGen MainMatrixGen

MainVectorGen: Reader.o VectorGen.o MainVectorGen.o
	g++ $^ -o MainVectorGen

VectorGen.o: $(VectorGen).C $(VectorGen).h
	g++ -I ../../Matrix/Utilities -c $(VectorGen).C

MainVectorGen.o: MainVectorGen.C
	g++ -I ../Utilities -c MainVectorGen.C

MainMatrixGen: Reader.o MatrixGen.o MainMatrixGen.o
	g++ $^ -o MainMatrixGen

MatrixGen.o: $(MatrixGen).C $(MatrixGen).h
	g++ -I ../../Matrix/Utilities -c $(MatrixGen).C

MainMatrixGen.o: MainMatrixGen.C
	g++ -I ../Utilities -c MainMatrixGen.C

Reader.o: $(Reader).C $(Reader).h
	g++ -c $(Reader).C


clean:
	rm -f *.o MainMatrixGen MainVectorGen

exe:
	./MainVectorGen vector.data 3 rand 1
	./MainMatrixGen matrix.data 3 rand 1 vector.data
