%include "Help.inc"

section .data
    write:   equ  4
    exit:    equ  1
    stderr:  equ  2

    nl:          db   10,0
    nl_len:      equ  $-nl

    help:        db   'Help:',10,10,0
    help_len:    equ  $-help

    prog:        db   'hidemsg "msg" -f <image> -o <output>',10,0
    prog_len:    equ  $-prog

    image:       db   9,'<image> = path to the image',10,0
    image_len:   equ  $-image

    output:      db   9,'<output> = name of the file where the output will be saved',10,10,0
    output_len:  equ  $-output

    note:        db   'Remember that the size of the message include the \0 character',10,0
    note_len:    equ  $-note

    note2:       db   'and must be smaller than the size of the image multiplied by 3',10,10,0
    note2_len:   equ  $-note2

section .text

_help:
    mov eax, write
    mov ebx, stderr
    mov ecx, nl
    mov edx, nl_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, help
    mov edx, help_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, prog
    mov edx, prog_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, image
    mov edx, image_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, output
    mov edx, output_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, note
    mov edx, note_len
    int 0x80

    mov eax, write
    mov ebx, stderr
    mov ecx, note2
    mov edx, note2_len
    int 0x80

_endErr:
    mov eax, exit
    mov ebx, 1
    int 0x80
