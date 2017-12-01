startHi (x:y:_)
    | x == 'h' && y == 'i' = True
    | otherwise = False

main = print ( startHi "hi" )
