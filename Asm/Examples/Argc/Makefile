Argc: Argc.o
	ld -m elf32_x86_64 Argc.o -o Argc

Argc.o: Argc.asm
	nasm -f elfx32 Argc.asm

clean:
	rm -f *.o Argc

exe:
	./Argc
