param=Beret.java \
	  Builder.java \
	  Cap.java \
	  Director.java \
	  Vest.java \
	  BuilderJacketBeret.java \
	  BuilderVestBeret.java \
	  Clothe.java \
	  Hat.java \
	  BuilderJacketCap.java \
	  BuilderVestCap.java \
	  Coat.java \
	  Jacket.java \
	  Outfit.java

MainBuilder.class: MainBuilder.java $(param)
	javac $<

clean:
	rm -f *.class

exe:
	java MainBuilder
