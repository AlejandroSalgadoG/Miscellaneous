arrayCount9 x = length [ n | n <- x , n == 9 ]

main = print (arrayCount9 [1,2,3,9,2])
