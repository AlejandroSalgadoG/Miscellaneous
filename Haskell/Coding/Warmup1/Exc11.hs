everyNth str n = everyNth' str 0 n (valid str n)

everyNth' str s n val = if null str
                        then []
                        else if s `elem` val
                             then head str : everyNth' (tail str) (s+1) n val
                             else everyNth' (tail str) (s+1) n val


valid str n = take (length str) [0,n..]

main = print( everyNth "Miracle" 2 )
