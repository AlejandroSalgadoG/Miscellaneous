%include "Error.inc"

section .data
    write:   equ  4
    stderr:  equ  2

    gtArgc:         db   'ERROR: Too much arguments',10,0
    gtArgc_len:     equ  $-gtArgc

    lwArgc:         db   'ERROR: Not enough arguments',10,0
    lwArgc_len:     equ  $-lwArgc

    nullStr:        db   'ERROR: null string',10,0
    nullStr_len:    equ  $-nullStr

    badFFlag:       db   'ERROR: first flag not recognized',10,0
    badFFlag_len    equ  $-badFFlag

    noExist:        db   'ERROR: File not foud',10,0
    noExist_len:    equ  $-noExist

    badOFlag:       db   'ERROR: second flag not recognized',10,0
    badOFlag_len:   equ  $-badOFlag

    errCreat:       db   'ERROR: Can not create the output file',10,0
    errCreat_len:   equ  $-errCreat

    msgToLong:      db   'ERROR: Message is to long',10,0
    msgToLong_len:  equ  $-msgToLong

section .text

_gtArgc:
    mov eax, write
    mov ebx, stderr
    mov ecx, gtArgc
    mov edx, gtArgc_len
    int 0x80
    jmp _help

_lwArgc:
    mov eax, write
    mov ebx, stderr
    mov ecx, lwArgc
    mov edx, lwArgc_len
    int 0x80
    jmp _help

_nullStr:
    mov eax, write
    mov ebx, stderr
    mov ecx, nullStr
    mov edx, nullStr_len
    int 0x80
    jmp _help

_badFFlag:
    mov eax, write
    mov ebx, stderr
    mov ecx, badFFlag
    mov edx, badFFlag_len
    int 0x80
    jmp _help

_noExist:
    mov eax, write
    mov ebx, stderr
    mov ecx, noExist
    mov edx, noExist_len
    int 0x80
    jmp _help

_badOFlag:
    mov eax, write
    mov ebx, stderr
    mov ecx, badOFlag
    mov edx, badFFlag_len
    int 0x80
    jmp _help

_errCreat:
    mov eax, write
    mov ebx, stderr
    mov ecx, errCreat
    mov edx, errCreat_len
    int 0x80
    jmp _help

_msgToLong:
    mov eax, write
    mov ebx, stderr
    mov ecx, msgToLong
    mov edx, msgToLong_len
    int 0x80
    jmp _help
