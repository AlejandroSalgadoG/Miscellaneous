inc=./Include

mainParam=Write.o \
		  Call.o \
		  CellPhone.o \
		  MainAdapter.o

mainAdapterParam=$(inc)/Write.h \
				 $(inc)/Call.h

writeParam=$(inc)/Comunication.h \
		   $(inc)/Write.h

callParam=$(inc)/Comunication.h \
		  $(inc)/Call.h \
		  $(inc)/CellPhone.h

MainAdapter: $(mainParam)
	g++ $^ -o $@

MainAdapter.o: MainAdapter.C $(mainAdapterParam)
	g++ -I$(inc) -c $<

Write.o: Write.C $(writeParam)
	g++ -I$(inc) -c $<

Call.o: Call.C $(callParam)
	g++ -I$(inc) -c $<

CellPhone.o: CellPhone.C $(inc)/CellPhone.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainAdapter

exe:
	./MainAdapter
