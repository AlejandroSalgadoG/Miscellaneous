main = print(automata "0.15 ")

automata :: String -> Bool
automata x = q0 x

q0 :: String -> Bool
q0 (x:xs)
    | x == '0' = q1 xs
    | x >= '1' && x <= '9' = q2 xs
    | otherwise = qerr xs

q1 :: String -> Bool
q1 (x:xs)
    | x == '.' = q3 xs
    | otherwise = qerr xs

q2 :: String -> Bool
q2 (x:xs)
    | x == '.' = q3 xs
    | x >= '1' && x <= '9' = q2 xs
    | otherwise = qerr xs

q3 :: String -> Bool
q3 (x:xs)
    | x >= '1' && x <= '9' = q4 xs
    | otherwise = qerr xs

q4 :: String -> Bool
q4 (x:xs)
    | x == ' ' = True
    | x >= '1' && x <= '9' = q4 xs
    | otherwise = qerr xs

qerr :: String -> Bool
qerr (x:xs)
    | x == ' ' = False
    | otherwise = qerr xs
