icyHot x y
    | x < 0 && y > 100 = True
    | y < 0 && x > 100 = True
    | otherwise = False

main = print ( icyHot 101 (-1) )
