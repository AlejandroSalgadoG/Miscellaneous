CondVar: CondVar.o
	g++ -pthread CondVar.o -o CondVar

CondVar.o: CondVar.C CondVar.h
	g++ -c CondVar.C

clean:
	rm -f *.o CondVar

exe:
	./CondVar
