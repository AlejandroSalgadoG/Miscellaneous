word = "ana"

palindrome x = palindrome' x x

palindrome' x y = if null x
                  then True
                  else if (head x) == (last y)
                       then palindrome' (tail x) (init y)
                       else False

main = print( palindrome word )
