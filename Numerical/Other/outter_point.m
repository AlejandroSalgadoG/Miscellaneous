function x = outter_point(fun, res, ro, mu, x_0, e)
    disp('Outter point output: ro, x_i, x_(i+1), distance, z')

    dist = e+1;
    
    while dist > e
        obj_fun = @(x) fun(x) + (ro/2) * res(x)^2;
        
        x = fminsearch(obj_fun, x_0);
        dist = sqrt(sum((x_0 - x) .^ 2));
        z = fun(x);
    
        output = sprintf('%f, (%f %f), (%f %f), %f, %f', ro, x_0, x, dist, z);
        disp(output)
        
        ro = ro * mu;
        x_0 = x;
    end
end