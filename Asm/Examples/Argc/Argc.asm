segment .text
    global _start

_start:
    mov ecx, esp
    add byte [esp], '0'

    mov eax, 4
    mov ebx, 1
    mov edx, 1
    int 0x80

    mov eax, 1
    mov ebx, 0

    int 0x80
