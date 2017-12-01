section .data
    file:   db      'file.txt',0

section .bss
    fd:     resb    4
    buff:   resb    3

section .text
    global _start

_start:

    mov eax, 5
    mov ebx, file
    mov ecx, 0

    int 80h

    mov ebx, eax

    mov eax, 3
    mov ecx, buff
    mov edx, 3

    int 80h

    mov eax, 4
    mov ebx, 1
    mov ecx, buff
    mov edx, 4

    int 80h

    mov eax, 6
    mov ebx, [fd]

    int 80h

    mov eax, 1
    mov ebx, 0
    int 80h
