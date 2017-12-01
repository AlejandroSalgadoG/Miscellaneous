max1020 x y
    | frst && scnd = max x y
    | frst = x
    | scnd = y
    | otherwise = 0
    where frst = x >= 10 && x <= 20
          scnd = y >= 10 && y <= 20

main = print( max1020 21 20)
