sumDouble x y
    | x == y = (x+y) * 2
    | otherwise = x+y

main = print ( sumDouble 2 2 )
