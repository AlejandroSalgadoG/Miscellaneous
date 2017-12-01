Param: Param.o
	ld -m elf32_x86_64 Param.o -o Param

Param.o: Param.asm
	nasm -f elfx32 Param.asm

clean:
	rm -f *.o Param

exe:
	./Param
