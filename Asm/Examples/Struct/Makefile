Struct: Struct.o
	ld -m elf32_x86_64 Struct.o -o Struct

Struct.o: Struct.asm Data.inc
	nasm -g -f elfx32 Struct.asm

clean:
	rm -f *.o Struct

gdb:
	make
	gdb -q ./Struct

exe:
	./Struct
