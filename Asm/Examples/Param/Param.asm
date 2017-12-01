section .data
    enter:   db   10
    nameSz:  equ  7

section .text
    global _start

_start:

    mov eax, 4
    mov ebx, 1
    mov ecx, esp
    add byte [ecx], '0'
    mov edx, 1
    int 0x80

    mov eax, 4
    mov ebx, 1
    mov ecx, enter
    mov edx, 1
    int 0x80

    mov eax, 4
    mov ebx, 1
    mov ecx, [esp+4]
    mov edx, nameSz
    int 0x80

    mov eax, 4
    mov ebx, 1
    mov ecx, enter
    mov edx, 1
    int 0x80

    mov eax, 4
    mov ebx, 1
    mov ecx, [esp+8]
    mov edx, 1
    int 0x80

    mov eax, 1
    mov ebx, 0

    int 0x80
