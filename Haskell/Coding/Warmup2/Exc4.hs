array123 x
    | length x < 3 = False
    | frst == 1 && scnd == 2 && thrd == 3 = True
    | otherwise = array123 (tail x)
    where frst = x !! 0
          scnd = x !! 1
          thrd = x !! 2
          
main = print ( array123 [0,1,2,3,4])
