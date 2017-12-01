stringTimes str 0 = []
stringTimes str c = str ++ stringTimes str (c-1)

main = print ( stringTimes "a" 3 )
