%include "HideMsg.inc"

section .data
    write:   equ  4
    exit:    equ  1
    stdout:  equ  1

    mask:  db  0x80

section .bss
    msgSz:  resd 1

section .text

_start:
    jmp _evalArgs

_saveMsgSz:
    mov ebx, 8
    div ebx
    mov [msgSz], eax
    jmp _savePnt

_getFstArg:
    mov esi, [esp+8] ; get first argument

_readChar:
    mov al, byte [esi]

_nextBit:
    test al, [mask]
    jz _readByteHope0
    jnz _readByteHope1

_decMask:
    mov eax, 0
    mov al, [mask]
    mov bl, 2
    div bl
    mov [mask], al
    cmp al, 0x00
    jne _readChar

_setMask:
    mov byte [mask], 0x80

_nextByte:
    inc esi
    mov al, byte [esi]
    dec dword [msgSz]
    cmp dword [msgSz], 0
    jne _nextBit
    je _writeOpt

_endOk:
    mov eax, exit
    mov ebx, 0
    int 0x80
