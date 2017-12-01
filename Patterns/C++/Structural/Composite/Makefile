inc=./Include

mainParam=Worker.o \
		  Assistant.o \
          Employee.o \
		  Boss.o \
          MainComposite.o

mainCompositeParam=Assistant.o \
				   Employee.o \
				   Boss.o \
				   Worker.o

MainComposite: $(mainParam)
	g++ $^ -o $@

MainComposite.o: MainComposite.C $(mainCompositeParam)
	g++ -I$(inc) -c $<

Boss.o: Boss.C $(inc)/Boss.h $(inc)/Worker.h
	g++ -I$(inc) -std=c++11 -c $<

Employee.o: Employee.C $(inc)/Employee.h $(inc)/Worker.h
	g++ -I$(inc) -c $<

Assistant.o: Assistant.C $(inc)/Assistant.h $(inc)/Worker.h
	g++ -I$(inc) -c $<

Worker.o: Worker.C $(inc)/Worker.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainComposite

exe:
	./MainComposite
