close10 x y = let differ1 = abs (10 - x)
                  differ2 = abs (10 - y)
              in if differ1 == differ2 then 0
                 else if differ1 > differ2 then y else x

main = print( close10 3 3 )
