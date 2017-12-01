data Person = Person  [Char] deriving Show

names = ["Alejo", "Ximena", "Laura"]

mapPerson = map (Person) names

main = print $ map (\(Person x) -> x ++ " Salgado") mapPerson
