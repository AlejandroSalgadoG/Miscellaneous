in1020 x y
    | frst || scnd = True
    | otherwise = False
    where frst = x >= 10 && x <= 20
          scnd = y >= 10 && y <= 20

main = print (in1020 9 21)
