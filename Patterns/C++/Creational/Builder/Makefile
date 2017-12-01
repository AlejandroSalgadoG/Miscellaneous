inc=./Include
cpp=-std=c++11

mainParam=Jacket.o \
		  Vest.o \
		  Beret.o \
		  Cap.o \
		  Outfit.o \
		  Director.o \
		  BuilderJacketBeret.o \
		  BuilderJacketCap.o \
		  BuilderVestBeret.o \
		  BuilderVestCap.o \
		  Clothe.o \
	      MainBuilder.o

mainBuilderParam=Director.o \
				 BuilderJacketBeret.o \
		  		 BuilderJacketCap.o \
		  		 BuilderVestBeret.o \
		  		 BuilderVestCap.o \
				 $(inc)/Builder.h

jacketBeretParam=Outfit.o \
				 Jacket.o \
				 Beret.o \
				 $(inc)/Builder.h

jacketCapParam=Outfit.o \
			   Jacket.o \
			   Cap.o \
			   $(inc)/Builder.h

vestBeretParam=Outfit.o \
			   Vest.o \
			   Beret.o \
			   $(inc)/Builder.h

vestCapParam=Outfit.o \
			 Vest.o \
			 Cap.o \
			 $(inc)/Builder.h

MainBuilder: $(mainParam)
	g++ $^ -o $@

MainBuilder.o: MainBuilder.C $(mainBuilderParam)
	g++ -I$(inc) -c $<

Director.o: Director.C $(inc)/Builder.h
	g++ -I$(inc) -c $<

BuilderJacketBeret.o: BuilderJacketBeret.C $(jacketBeretParam)
	g++ -I$(inc) -c $<

BuilderJacketCap.o: BuilderJacketCap.C $(jacketCapParam)
	g++ -I$(inc) -c $<

BuilderVestBeret.o: BuilderVestBeret.C $(vestBeretParam)
	g++ -I$(inc) -c $<

BuilderVestCap.o: BuilderVestCap.C $(vestCapParam)
	g++ -I$(inc) -c $<

Outfit.o: Outfit.C $(inc)/Outfit.h $(inc)/Clothe.h
	g++ $(cpp) -I$(inc) -c $<

Jacket.o: Jacket.C $(inc)/Jacket.h $(inc)/Coat.h
	g++ -I$(inc) -c $<

Vest.o: Vest.C $(inc)/Vest.h $(inc)/Coat.h
	g++ -I$(inc) -c $<

Beret.o: Beret.C $(inc)/Beret.h $(inc)/Hat.h
	g++ -I$(inc) -c $<

Cap.o: Cap.C $(inc)/Cap.h $(inc)/Hat.h
	g++ -I$(inc) -c $<

Clothe.o: Clothe.C $(inc)/Clothe.h
	g++ -I$(inc) -c $<

clean:
	rm -f *.o MainBuilder

exe:
	./MainBuilder
