Read: Read.o
	ld -m elf32_x86_64 Read.o -o Read

Read.o: Read.asm
	nasm -f elfx32 Read.asm

clean:
	rm -f *.o Read

exe:
	./Read
