list = [1,2,3,4,5]

duplicate x = if null x
              then []
              else conc (head x) ++ duplicate (tail x)

conc x = x:x:[]

main = print(duplicate list)
