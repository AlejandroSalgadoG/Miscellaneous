Write: Write.o
	ld -m elf32_x86_64 Write.o -o Write

Write.o: Write.asm
	nasm -g -f elfx32 Write.asm

clean:
	rm -f *.o Write

gdb:
	make
	gdb -q ./Write

exe:
	./Write
