%: %.o
	ghc $< -o $@

%.o: %.hs
	ghc -c $<

clean:
	rm -f *.o *.hi Exc*[0-9]
