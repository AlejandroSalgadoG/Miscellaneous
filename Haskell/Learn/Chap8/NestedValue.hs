data Person = Person [Char] Lastname deriving Show
data Lastname = Lastname [Char] deriving Show

names = ["Alejo", "Juan"]
lastnames = ["Salgado", "Cardona"]

mapLastname = map Lastname lastnames
mapNames = map Person names

mapPerson [] _ = []
mapPerson (f:fs) (x:xs) = f x : mapPerson fs xs

getName (Person name (Lastname last)) = name ++ " " ++ last

main = print $ map getName persons
       where persons = mapPerson mapNames mapLastname
