Reader=../Utilities/Reader

MainGauss: Reader.o Gauss.o MainGauss.o
	g++ $^ -o MainGauss

Reader.o: $(Reader).C $(Reader).h
	g++ -c $(Reader).C

Gauss.o: Gauss.C Gauss.h
	g++ -c Gauss.C

MainGauss.o: MainGauss.C
	g++ -I../Utilities -c MainGauss.C

clean:
	rm -f *.o MainGauss

exe:
	./MainGauss 3 matrix.data
