frontBack x = (last x :[]) ++ tail (init x) ++ (head x :[])

main = print ( frontBack "hola" )
