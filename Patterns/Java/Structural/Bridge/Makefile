mainParam=Animal.java \
		  Cat.java \
		  Dog.java \
		  Jump.java \
		  Movement.java \
		  Run.java

MainBridge.class: MainBridge.java
	javac $<

clean:
	rm -f *.class MainBridge.class

exe:
	java MainBridge
