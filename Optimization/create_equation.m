function restriction = create_equation(vars, rest_vars, rest_vals)
    %height = 1;
    width = 2;

    var_num = size(vars, width);
    rest_vars_num = size(rest_vars, width);
    
    index = 1:var_num;
    var_index = containers.Map(vars,index);
    
    restriction = zeros(1, var_num);

    for idx = 1:rest_vars_num
        var = rest_vars(idx);
        var_idx = var_index(var{1});
        restriction(var_idx) = rest_vals(idx);
    end
end