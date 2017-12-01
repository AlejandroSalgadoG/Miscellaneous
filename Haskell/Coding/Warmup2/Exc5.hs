conc x y
    | null x = y
    | null y = x
    | otherwise = head x : head y : [] ++ conc (tail x) (tail y)

altPairs x = let len = (length x) - 1
                 frst = [n | n <- [0,4..len]]
                 scnd = [m | m <- [1,5..len]]
                 list = conc frst scnd
             in altPairs' x list

altPairs' x [] = []
altPairs' x list = let frst = head list
                       rest = tail list
                   in (x !! frst) : altPairs' x rest

main = print (altPairs "Holahola")
