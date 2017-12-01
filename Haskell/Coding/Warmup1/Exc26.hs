front22 [] = []
front22 (x:[]) = x:x:x:[]
front22 all@(x:y:_) = x:y:[] ++ all ++ x:y:[]

main = print( front22 "Ha")
