vars = {'wr','we','wa','wc','n1','n2','n3','n4','n5','n6','p1','p2','p3','p4','p5','p6'};

obj_fun = create_equation(vars, {'n1','n2','n3','n4','n5','n6','p1','p2','p3','p4','p5','p6'}, ...
                                [  1    1    1    1    1    1    1    1    1    1    1    1]);

r1 = create_equation(vars, {'wr', 'we', 'n1', 'p1'}, ...
                           [ 1   -0.8    1     -1 ]);
                       
r2 = create_equation(vars, {'wr', 'wa', 'n2', 'p2'}, ...
                           [ 1   -1.1    1     -1 ]);
                       
r3 = create_equation(vars, {'wr', 'wc', 'n3', 'p3'}, ...
                           [ 1   -1.5    1     -1 ]);
                       
r4 = create_equation(vars, {'we', 'wa', 'n4', 'p4'}, ...
                           [ 1   -1.2    1     -1 ]);
                       
r5 = create_equation(vars, {'we', 'wc', 'n4', 'p4'}, ...
                           [ 1   -1.3    1     -1 ]);
                       
r6 = create_equation(vars, {'wa', 'wc', 'n4', 'p4'}, ...
                           [ 1   -0.95    1    -1 ]);
                       
r_eq = create_equation(vars, {'wr', 'we', 'wa', 'wc'}, ...
                             [ 1      1     1     1 ]);
                       
A = [];
b = [];
                         
Aeq = [ r1; r2; r3; r4; r5; r6; r_eq ];                       
beq = [ 0; 0; 0; 0; 0; 0; 1 ];

lb = zeros(16,1);

optimal_values = linprog(obj_fun,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
        