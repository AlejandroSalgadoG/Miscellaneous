list = [1,2,3,4,5]

slice i x c = slice' 0 i x c

slice' s i x c = if s < i
                 then slice' (s+1) i (tail x) c
                 else if c == 0
                      then []
                      else head x : slice' s i (tail x) (c-1)

main = print( slice 1 list 3 )
