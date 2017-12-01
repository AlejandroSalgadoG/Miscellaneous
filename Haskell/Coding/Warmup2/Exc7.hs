frontTimes x y 
    | y == 0 = []
    | length x < 4 = x ++ frontTimes x (y-1)
    | otherwise = frst : scnd : thrd : [] ++ frontTimes x (y-1)
    where frst = x !! 0
          scnd = x !! 1
          thrd = x !! 2
          


main = print (frontTimes "abcd" 2)
