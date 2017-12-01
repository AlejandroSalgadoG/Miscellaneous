section .data
    file:   db      'text.txt',0
    msg:    db      'The file',10
    msgL:   equ     9

section .bss
    fd:     resb    4

section .text
        global _start

_start:
        mov eax, 8
        mov ebx, file
        mov ecx, 644O
        int 80h

        mov [fd], eax

        mov eax, 4
        mov ebx, [fd],
        mov ecx, msg
        mov edx, msgL
        int 80h

        mov eax, 6
        mov ebx, [fd]
        int 80h

        mov eax, 1
        mov ebx, 0
        int 80h
