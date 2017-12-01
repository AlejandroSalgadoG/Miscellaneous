Write: Write.o
	ld -m elf32_x86_64 Write.o -o Write

Write.o: Write.asm
	nasm -f elfx32 Write.asm

clean:
	rm -f *.o text.txt Write

exe:
	./Write
