%include "File.inc"
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

section .bss
    endData:   resd  1
    startOpt:  resd  1
    inputP:    resd  1
    input:     resd  1
    outputP:   resd  1
    output:    resd  1
    statSz:    resd  1
    statP:     resb  Stat_size

section .text

_setIpt:
    mov eax, [esp+8]
    mov [input], eax
    ret

_setOpt:
    mov eax, [esp+8]
    mov [output], eax
    ret

_allocFiles:
    jmp _getFileSz

_getFileSz:
    mov eax, stat
    mov ebx, [esp+16]
    mov ecx, statP
    int 0x80

    mov eax, dword [statP+Stat.st_size]
    mov [statSz], eax

_getDataLim:
    mov eax, brk
    mov ebx, boundary
    int 0x80
    mov [endData], eax

_movDataLimIpt:
    mov eax, [endData]
    mov [inputP], eax
    mov eax, brk
    mov ebx, [endData]
    add ebx, dword [statSz]
    mov [outputP], ebx
    mov [startOpt], ebx
    int 0x80

_movDataLimOpt:
    mov eax, brk
    mov ebx, [outputP]
    add ebx, dword [statSz]
    int 0x80

_saveIpt:
    mov eax, read
    mov ebx, [input]
    mov ecx, [inputP]
    mov edx, dword [statSz]
    int 0x80

    mov ecx, 0  ; set counter to 0

_cpHeadToOpt:
    dec dword [statSz]

    mov eax, [inputP]
    mov bl, [eax]

    mov eax, [outputP]
    mov [eax], bl

    add dword [inputP], 1
    add dword [outputP], 1

    cmp bl, 0x0a
    jne _cpHeadToOpt

    inc ecx
    cmp ecx, 3
    jne _cpHeadToOpt

_saveImgSz:
    mov eax, [statSz]
    jmp _evalMsgSz

_savePnt:
    sub esp, 4
    mov eax, [inputP]
    mov [esp], eax

    sub esp, 4
    mov eax, [outputP]
    mov [esp], eax

_cpIptToOpt:
    mov eax, [inputP]
    mov bl, [eax]

    mov eax, [outputP]
    mov [eax], bl

    add dword [inputP], 1
    add dword [outputP], 1

    dec dword [statSz]
    cmp dword [statSz], 0
    jne _cpIptToOpt

    mov eax, [esp]
    mov [outputP], eax
    add esp, 4

    mov eax, [esp]
    mov [inputP], eax
    add esp, 4

    jmp _getFstArg

_readByteHope0:
    mov eax, [inputP]
    mov bl, [eax]

    test bl, 0x01
    jz _incPnt

_bitOff:
    mov eax, [outputP]
    dec bl
    mov [eax], bl
    jmp _incPnt

_readByteHope1:
    mov eax, [inputP]
    mov bl, [eax]

    test bl, 0x01
    jnz _incPnt

_bitOn:
    mov eax, [outputP]
    inc bl
    mov [eax], bl

_incPnt:
    add dword [inputP], 1
    add dword [outputP], 1
    jmp _decMask

_writeOpt:
    mov eax, write
    mov ebx, [output]
    mov ecx, [startOpt]
    mov edx, dword [statP+Stat.st_size]
    int 0x80

_closeIpt:
    mov eax, close
    mov ebx, [input]

_closeOpt:
    mov eax, close
    mov ebx, [output]
    call _retDataLim
    jmp _endOk

_retDataLim:
    mov eax, brk
    mov ebx, [endData]
    int 0x80
    ret
