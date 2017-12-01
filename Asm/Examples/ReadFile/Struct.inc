%ifndef STRUCT_INC
%define STRUCT_INC

struc Stat
    .st_dev:     resd  1  ; Id of device containing the file
    .st_ino:     resd  1  ; Inode number
    .st_mode:    resw  1  ; Protection
    .st_nlink    resw  1  ; number of hard links
    .st_uid      resw  1  ; user ID of owner
    .st_gid      resw  1  ; group ID of owner
    .st_rdev     resd  1  ; device ID (if special file)
    .st_size     resd  1  ; total size, in bytes
    .st_blksize  resd  1  ; blocksize for filesystem I/O
    .st_blocks   resd  1  ; number of 512B blocks allocated
    .st_atime    resd  1  ; time of last access
    .st_mtime    resd  1  ; time of last modification
    .st_ctime    resd  1  ; time of last status change
endstruc

%endif
