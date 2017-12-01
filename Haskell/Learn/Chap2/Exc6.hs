list = [1,2,3,4,5,3]

dropall x y = if null y
            then []
            else if x == head y
                 then dropall x (tail y)
                 else head y : dropall x (tail y)

dropk i x = dropk' i x 0

dropk' i x c = if null x
               then []
               else if i == c
                    then dropk' i (tail x) (c+1)
                    else head x : dropk' i (tail x) (c+1)

main = print( dropk 3 list )
