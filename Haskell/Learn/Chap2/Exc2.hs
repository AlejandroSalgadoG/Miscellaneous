list = [1,2,3,4,5]

find x y = if null y
           then False
           else if  x == (head y)
                then True
                else find x (tail y)

main = print(find 3 list)
