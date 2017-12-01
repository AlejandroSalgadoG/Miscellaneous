%include "variables.inc"
%include "variables.inc"

section .text
    global _start

_start:

    mov eax, 4
    mov ebx, 1
    mov ecx, msg
    mov edx, 21
    int 0x80

_end:
    mov eax, 1
    mov ebx, 0
    int 0x80
