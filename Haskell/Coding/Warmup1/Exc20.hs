in3050 x y
    | frst || scnd = True
    | otherwise = False
    where frst = (x >= 30 && x <= 40) && (y >= 30 && y <= 40)
          scnd = (x >= 40 && x <= 50) && (y >= 40 && y <= 50)

main = print (in3050 40 50)
