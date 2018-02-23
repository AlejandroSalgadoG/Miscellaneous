function x = coordinates(fun, x_0, e)  
    disp('Coordinates output: x_i, x_(i+1), distance, z')

    syms lambda
    dist = e+1;
    iter = 0;
    
    while dist > e
        v = get_vector(iter);
    
        sym_obj_fun = fun(x_0 + lambda*v);
        obj_fun = matlabFunction(sym_obj_fun);
        lambda_val = fminsearch(obj_fun, 0);
    
        x = x_0 + lambda_val*v;
        dist = sqrt(sum((x_0 - x) .^ 2));
        z = fun(x);
    
        output = sprintf('(%f %f), (%f %f), %f, %f', x_0, x, dist, z);
        disp(output)
    
        x_0 = x;
        
        iter = iter + 1;
    end
end

function v = get_vector(iter)
    if mod(iter,2) == 0
        v = [1 0]; % horizontal
    else
        v = [0 1]; % vertical
    end
end