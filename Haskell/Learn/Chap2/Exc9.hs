list = [1,2,3,4,5]

rotate n x = rotate' 0 n x

rotate' s n x = if s == n
                then x
                else rotate' (s+1) n (tail x ++ head x : [])

main = print(rotate 5 list)
