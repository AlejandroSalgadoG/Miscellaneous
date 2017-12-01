posNeg x y False
    | (x * y) < 0 = True
    | otherwise = False


posNeg x y True
    | (x * y) > 0 = True
    | otherwise = False


main = print ( posNeg 1 (-1) False )
