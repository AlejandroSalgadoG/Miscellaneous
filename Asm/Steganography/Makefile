INC=Include

LD=ld
LD_FLAGS=-m elf32_x86_64

CXX=nasm
CXX_FLAGS=-I $(INC)/ -g -f elfx32

hidemsg: Help.o Error.o File.o Validation.o HideMsg.o
	$(LD) $(LD_FLAGS) $^ -o $@

Help.o: Help.asm $(INC)/Help.inc
	$(CXX) $(CXX_FLAGS) $<

File.o: File.asm $(INC)/File.inc
	$(CXX) $(CXX_FLAGS) $<

Error.o: Error.asm $(INC)/Error.inc
	$(CXX) $(CXX_FLAGS) $<

Validation.o: Validation.asm $(INC)/Validation.inc
	$(CXX) $(CXX_FLAGS) $<

HideMsg.o: HideMsg.asm $(INC)/HideMsg.inc
	$(CXX) $(CXX_FLAGS) $<

clean:
	rm -f *.o hidemsg output.ppm

gdb:
	make
	gdb -q ./hidemsg

exe:
	./hidemsg "hello" -f white.ppm -o output.ppm
	hexdump -Cv output.ppm
