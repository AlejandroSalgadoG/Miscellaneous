import Data.Char

endUp [] = []
endUp (x:[]) = toUpper x :[]
endUp (x:y:[]) = toUpper x : toUpper y :[]
endUp (x:y:z:[]) = toUpper x : toUpper y : toUpper z : []
endUp (s:x:y:z:[]) = s : toUpper x : toUpper y : toUpper z : []

main = print( endUp "hola")
