stringE x
    | ecount >= 1 && ecount <= 3 = True
    | otherwise = False
    where ecount = length [ e | e <- x, e == 'e' ]

main = print( stringE "eEeEe" )
