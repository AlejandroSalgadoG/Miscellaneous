noTriples x
    | length x < 3 = True
    | (frst == scnd) && (scnd == thrd) = False
    | otherwise = noTriples (tail x)
    where frst = x !! 0
          scnd = x !! 1
          thrd = x !! 2

main = print ( noTriples [1,1,1])
