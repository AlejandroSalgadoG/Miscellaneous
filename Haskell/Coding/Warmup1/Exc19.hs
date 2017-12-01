startOz [] = ""
startOz [x] = ""
startOz (x:y:_)
    | x == 'o' && y == 'z' = "oz"
    | x == 'o' = "o"
    | y == 'z' = "z"
    | otherwise = ""

main = print (startOz "oz")
