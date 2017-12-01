doubleX [] = False
doubleX (x:[]) = False
doubleX x
    | head x == 'x' && head (tail x) == 'x' = True
    | otherwise = doubleX (tail x)
   

main = print ( doubleX "" )
