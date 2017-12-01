list = [1,2,3,4,5]

insertElem x y p = insertElem' x y 0 (length y) p

insertElem' x y s l p = if s == (l+1)
                        then []
                        else if s == p
                             then x : insertElem' x y (s+1) l p
                             else head y : insertElem' x (tail y) (s+1) l p

main = print( insertElem 6 list 5 )
