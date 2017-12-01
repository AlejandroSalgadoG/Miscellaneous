Function: Function.o
	ld -m elf32_x86_64 Function.o -o Function

Function.o: Function.asm
	nasm -f elfx32 Function.asm

clean:
	rm -f *.o Function

exe:
	./Function
