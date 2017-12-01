stringMatch frst@(w:x:_) scnd@(y:z:_)
    | w == y && x == z = 1 + stringMatch (tail frst) (tail scnd) 

stringMatch x y = 0
          

main = print ( stringMatch "ab" "abc")
