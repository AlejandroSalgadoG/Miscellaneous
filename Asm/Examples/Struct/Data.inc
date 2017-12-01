%ifndef STRUCT
%define STRUCT

struc Msg
    .msg1:  resb  4
    .msg2:  resb  4
endstruc

segment .data
    msg: istruc Msg
             at Msg.msg1,  db 'ok',10,0
             at Msg.msg2,  db 'ko',10,0
         iend

segment .bss
    pointPlace:  resb  Msg_size

%endif
