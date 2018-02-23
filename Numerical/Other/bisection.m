function l = bisection(fun, inter, delta, e)
    disp('Bisection output: [low up], l_0/ln')

    low = inter(1);
    up = inter(2);
    
    L = up - low;
    
    while (up-low)/L > e
        x = (low + up)/2;
       
        yl = fun(x - delta);
        yu = fun(x + delta);
       
        if yl > yu
            low = x - delta;
        else 
            up = x + delta;
        end
        
        output = sprintf('[%f %f], %f', low, up, (up-low)/L);
        disp(output)
    end
    
    l = [low up];
end