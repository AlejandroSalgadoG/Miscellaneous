diff21 x
    | x <= 21 = abs (x - 21)
    | otherwise = abs (x - 21) * 2

main = print( diff21 50 )
