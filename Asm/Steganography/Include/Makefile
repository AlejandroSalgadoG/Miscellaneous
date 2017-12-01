Include: Include.o
	ld -m elf32_x86_64 Include.o -o Include

Include.o: Include.asm
	nasm -f elfx32 Include.asm

clean:
	rm -f *.o Include

exe:
	./Include
