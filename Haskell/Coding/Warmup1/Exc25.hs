front3 [] = []
front3 (x:[]) = x:x:x:[]
front3 (x:y:[]) = x:y:x:y:x:y:[]
front3 (x:y:z:_) = x:y:z:x:y:z:x:y:z:[]

main = print( front3 "abc")
