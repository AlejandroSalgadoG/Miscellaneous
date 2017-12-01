notString all@('n':'o':'t':_) = all
notString x = "not " ++ x

main = print ( notString "a" )
