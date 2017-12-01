last2 x
    | length x <= 3 = 0
    | otherwise = count patt str
    where  str = init (init x)
           patt = last (init x) : last x : []

count patt str
    | length str < 2 = 0
    | (take 2 str) == patt = 1 + count patt nStr
    | otherwise = count patt (tail str)
    where nStr = drop 2 str

main = print ( last2 "cdacdcdcd" )
