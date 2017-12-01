lastDigit x y
    | last1 == last2 = True
    | otherwise = False
    where last1 = last (show x)
          last2 = last (show y)

main = print ( lastDigit 18 77 )
