parrotTrouble x y
    | x == True && (y < 7 || y > 20) = True
    | otherwise = False

main = print ( parrotTrouble True 7 )
