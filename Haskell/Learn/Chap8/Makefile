%: %.o
	ghc $< -o $@

%.o: %.hs
	ghc -c $<

clean:
	rm -f *.o *.hi ValueConst NestedValue
