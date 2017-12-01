mainParam=Assistant.java \
          Boss.java \
		  Employee.java \
		  Worker.java

MainComposite.class: MainComposite.java $(mainParam)
	javac $<

clean:
	rm -f *.class 

exe:
	java MainComposite
