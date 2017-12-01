mixStart (_:x:y:_)
    | x == 'i' && y == 'x' = True
    | otherwise = False

main = print ( mixStart "mix" )
