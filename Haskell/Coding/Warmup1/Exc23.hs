makes10 10 _ = True
makes10 _ 10 = True
makes10 x y
    | x + y == 10 = True
    | otherwise = False

main = print ( makes10 2 10 )
