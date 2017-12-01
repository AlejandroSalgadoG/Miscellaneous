loneTeen x y = let frst = x >= 13 && x <= 19
                   scnd = y >= 13 && y <= 19
               in
                   if frst /= scnd then True
                   else False

main = print ( loneTeen 1 13 )
