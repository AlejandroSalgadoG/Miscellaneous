missingChar str p = missingChar' str p 0

missingChar' str p c = if null str
                       then []
                       else if p == c
                            then missingChar' (tail str) p (c+1)
                            else head str : missingChar' (tail str) p (c+1)

main = print ( missingChar "kitten" 0 )
