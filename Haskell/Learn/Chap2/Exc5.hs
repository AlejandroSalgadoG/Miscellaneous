list1 = [1,2,3,4,5]
list2 = ['a','b','c','d','e']

ziplike x y = if null x || null y
              then [] 
              else conc (head x) (head y) ++ ziplike (tail x) (tail y)

conc x y = (x, y) : []

main = print( ziplike list1 list2 )
