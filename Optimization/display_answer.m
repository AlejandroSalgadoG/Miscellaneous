function display_answer(vars, opt_value)
    %height =1;
    width = 2;

    var_num = size(vars, width);

    for idx=1:var_num
        var = vars(idx);
        fprintf('%s = %.2f\n', var{1}, opt_value(idx))
    end
end