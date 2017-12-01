Hello: Hello.o
	ld -m elf32_x86_64 Hello.o -o Hello

Hello.o: Hello.asm
	nasm -f elfx32 Hello.asm

clean:
	rm -f *.o Hello

exe:
	./Hello
