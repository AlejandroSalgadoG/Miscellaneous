section .data
    msg:  db  'function',10

section .text
    global _start

print:
    mov eax, 4
    mov ebx, 1
    mov ecx, [esp+8]
    mov edx, 9
    int 80h
    ret

_start:
    push msg
    call print

    mov eax, 1
    mov ebx, 0
    int 80h
