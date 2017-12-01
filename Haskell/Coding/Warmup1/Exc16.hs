or35 x = let three = x `mod` 3
             five = x `mod` 5
         in three == 0 || five == 0

main = print ( or35 11 )
