function display_goals(restrictions, goals, opt_values)
    height =1;
    %width = 2;

    rest_num = size(restrictions, height);
    
    for idx = 1:rest_num
       rest = restrictions(idx,:);
       rest_val = dot(rest, opt_values);
       fprintf('rest %d, %.2f/%.2f = %.2f\n',idx, rest_val, goals(idx), rest_val/goals(idx))
    end
end