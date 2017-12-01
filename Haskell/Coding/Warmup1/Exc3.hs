nearHoundred x
    | diff1 <= 10 = True
    | diff2 <= 10 = True
    | otherwise = False
    where diff1 = abs (100 - x)
          diff2 = abs (200 - x)

main = print( nearHoundred 189 )
