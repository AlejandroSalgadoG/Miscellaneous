hasTeen x y z
    | x >= 13 && x <= 19 = True
    | y >= 13 && y <= 19 = True
    | z >= 13 && z <= 19 = True
    | otherwise = False

main = print ( hasTeen 13 2 3 )
