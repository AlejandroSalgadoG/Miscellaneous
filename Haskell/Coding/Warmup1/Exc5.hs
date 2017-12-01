backArround x = let lastest = last x : []
                in lastest ++ x ++ lastest

main = print( backArround "cat" )
