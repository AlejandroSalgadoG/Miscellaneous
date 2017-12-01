%include "Struct.inc"

section .data
    stat:      equ  106
    brk:       equ  45
    open:      equ  5
    read:      equ  3
    write:     equ  4
    close:     equ  6
    exit:      equ  1
    boundary:  equ  0
    O_RDONLY:  equ  0
    stdout:    equ  1

    file:  db  'white.ppm',0

section .bss
    statP:    resb  Stat_size
    endData:  resd  1
    fd:       resd  1

section .text
    global _start

_start:

    mov eax, stat
    mov ebx, file
    mov ecx, statP
    int 0x80

    mov eax, brk
    mov ebx, boundary
    int 0x80

    mov [endData], eax

    mov eax, brk
    mov ebx, [endData]
    add ebx, dword [statP+Stat.st_size]
    int 0x80

    mov eax, open
    mov ebx, file
    mov ecx, O_RDONLY
    int 0x80

    mov [fd], eax

    mov eax, read
    mov ebx, [fd]
    mov ecx, [endData]
    mov edx, dword [statP+Stat.st_size]
    int 0x80

    mov eax, write
    mov ebx, stdout
    mov ecx, [endData]
    mov edx, dword [statP+Stat.st_size]
    int 0x80

    mov eax, close
    mov ebx, [fd]
    int 0x80

    mov eax, brk
    mov ebx, [endData]
    int 0x80

_end:
    mov eax, exit
    mov ebx, 0
    int 0x80
