%include "Validation.inc"

section .data
    open:       equ  5
    creat:      equ  8
    write:      equ  4
    O_RDONLY:   equ  0
    stdout:     equ  1

    ok:      db   'All arguments are ok',10,0
    ok_len:  equ  $-ok

section .text

_evalArgs:
    jmp _numArgs

_numArgs:
    mov eax, [esp]  ; get argc
    cmp eax, 6      ; argc == 6?
    jg _gtArgc      ; jump if greater
    jl _lwArgc      ; jump if lower

_fstArg:
    mov eax, [esp+8]   ; get first argument
    cmp byte [eax], 0  ; null string ?
    je _nullStr        ; jump if equal

_secArg:
    mov eax, [esp+12]     ; get second argument
    cmp word [eax], '-f'  ; arg == -f?
    jne _badFFlag         ; jump if not equal

_trdArg:
    mov eax, open
    mov ebx, [esp+16]  ; get third argument (image)
    mov ecx, O_RDONLY
    int 0x80

    sub esp, 4
    mov [esp], eax  ; save the file descriptor in the stack
    call _setIpt  ; call function to save the fd in the main file
    add esp, 4

    cmp eax, 0         ; file was found?
    jl _noExist

_forArg:
    mov eax, [esp+20]     ; get fourth argument
    cmp word [eax], '-o'  ; arg == -o?
    jne _badOFlag

_fifArg:
    mov eax, creat         ; creat
    mov ebx, [esp+24]  ; get fifth argument (output)
    mov ecx, 664O      ; rw-rw-r--
    int 0x80

    sub esp, 4
    mov [esp], eax   ; save the file descriptor in the stack
    call _setOpt  ; call function to save the fd int the main file
    add esp, 4

    cmp eax, 0         ; file was created?
    jl _errCreat

_argsOk:
    mov eax, write
    mov ebx, stdout
    mov ecx, ok
    mov edx, ok_len
    int 0x80
    jmp _allocFiles

_evalMsgSz:
    mov ebx, eax
    mov eax, 0       ; counter in 0
    mov esi, [esp+8] ; mov the msg to esi

_getMsgSz:
    inc eax
    inc esi
    cmp byte [esi], 0
    jne _getMsgSz
    inc eax

    mov ecx, 8
    mul ecx

    cmp eax, ebx
    jle _saveMsgSz
    call _retDataLim
    jmp _msgToLong
