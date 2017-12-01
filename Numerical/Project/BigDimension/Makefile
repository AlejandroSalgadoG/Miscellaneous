Reader=../../Matrix/Utilities/Reader
Norm=../../Matrix/Utilities/Norm

MainBigJacobi: Reader.o Norm.o Info.o ReadBig.o BigJacobi.o MainBigJacobi.o
	g++ $^ -o MainBigJacobi

Reader.o: $(Reader).C $(Reader).h
	g++ -c $(Reader).C

Norm.o: $(Norm).C $(Norm).h
	g++ -c $(Norm).C

Info.o: Info.C Info.h
	g++ -c Info.C

ReadBig.o: ReadBig.C ReadBig.h
	g++ -c ReadBig.C

BigJacobi.o: BigJacobi.C BigJacobi.h
	g++ -I../../Matrix/Utilities -c BigJacobi.C

MainBigJacobi.o: MainBigJacobi.C
	g++ -I../../Matrix/Utilities -c MainBigJacobi.C

clean:
	rm -f *.o MainBigJacobi

exe:
	./MainBigJacobi matrix.data 45 0.0000008 0.000005 20 vector.data
