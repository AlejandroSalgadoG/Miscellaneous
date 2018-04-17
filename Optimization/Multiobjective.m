%% basic example
                 %4.27
f = [0 0 0 1 1 1 0 1 0 0 0 1 1];

intcon = [1 2 3];

A = [];
b = [];

Aeq = [ 1   0   0    1 0 0 0 0 -1  0  0  0  0;
        0   1   0    0 1 0 0 0  0 -1  0  0  0;
        0   0   1    0 0 1 0 0  0  0 -1  0  0;
       18  33  45.15 0 0 0 1 0  0  0  0 -1  0;
        4  7.5 10.5  0 0 0 0 1  0  0  0  0 -1];

beq = [5;
       10;
       15;
       1000;
       250];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];
   
intlinprog(f,intcon,A,b,Aeq,beq,lb)

%% Optimization without normalization
vars = {'x1','x2','x3','n1','n2','n3','n4','n5','p1','p2','p3','p4','p5'};

obj_fun = create_equation(vars, {'n1', 'n2', 'n3', 'p4', 'n5', 'p5'}, ...
                                [ 1     1     1     1     1     1 ]);

small_rest = create_equation(vars, {'x1', 'n1', 'p1'}, ...
                                   [ 1     1     -1 ]);

medium_rest = create_equation(vars, {'x2', 'n2', 'p2'}, ...
                                    [ 1     1     -1 ]);

large_rest = create_equation(vars, {'x3', 'n3', 'p3'}, ...
                                   [ 1     1     -1 ]);

budget_rest = create_equation(vars, { 'x1',  'x2',   'x3', 'n4', 'p4'}, ...
                                    [ 18000  33000  45150   1     -1 ]);

space_rest = create_equation(vars, { 'x1',  'x2', 'x3', 'n5', 'p5'}, ...
                                   [ 400    750   1050   1     -1 ]);

intcon = [1 2 3];

A = [];
b = [];

Aeq = [ small_rest;
        medium_rest;
        large_rest;
        budget_rest;
        space_rest ];

beq = [5;
       10;
       15;
       1000000;
       25000];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];

optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(:,1:3), beq, optimal_values(1:3));

%% Optimization with normalization
vars = {'x1','x2','x3','n1','n2','n3','n4','n5','p1','p2','p3','p4','p5'};

obj_fun = create_equation(vars, { 'n1', 'n2', 'n3',   'p4',      'n5',    'p5'}, ...
                                [ 1/5   1/10  1/15  1/1000000  1/25000  1/25000 ]);

small_rest = create_equation(vars, {'x1', 'n1', 'p1'}, ...
                                   [ 1     1     -1 ]);

medium_rest = create_equation(vars, {'x2', 'n2', 'p2'}, ...
                                    [ 1     1     -1 ]);

large_rest = create_equation(vars, {'x3', 'n3', 'p3'}, ...
                                   [ 1     1     -1 ]);

budget_rest = create_equation(vars, { 'x1',  'x2',   'x3', 'n4', 'p4'}, ...
                                    [ 18000  33000  45150   1     -1 ]);

space_rest = create_equation(vars, { 'x1',  'x2', 'x3', 'n5', 'p5'}, ...
                                   [ 400    750   1050   1     -1 ]);

intcon = [1 2 3];

A = [];
b = [];

Aeq = [ small_rest;
        medium_rest;
        large_rest;
        budget_rest;
        space_rest];

beq = [5;
       10;
       15;
       1000000;
       25000];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];

optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(:,1:3), beq, optimal_values(1:3));

%% Optimization with normalization and ponderation
vars = {'x1','x2','x3','n1','n2','n3','n4','n5','p1','p2','p3','p4','p5'};

obj_fun = create_equation(vars, { 'n1', 'n2', 'n3',   'p4',      'n5',   'p5' }, ...
                                [ 1/5   1/10  1/15  2/1000000  1/25000  1/25000 ])

small_rest = create_equation(vars, {'x1', 'n1', 'p1'}, ...
                                   [ 1     1     -1 ]);

medium_rest = create_equation(vars, {'x2', 'n2', 'p2'}, ...
                                    [ 1     1     -1 ]);

large_rest = create_equation(vars, {'x3', 'n3', 'p3'}, ...
                                   [ 1     1     -1 ]);

budget_rest = create_equation(vars, { 'x1',  'x2',   'x3', 'n4', 'p4'}, ...
                                    [ 18000  33000  45150   1     -1 ]);

space_rest = create_equation(vars, { 'x1',  'x2', 'x3', 'n5', 'p5'}, ...
                                   [ 400    750   1050   1     -1 ]);

intcon = [1 2 3];

A = [];
b = [];

Aeq = [ small_rest;
        medium_rest;
        large_rest;
        budget_rest;
        space_rest];

beq = [5;
       10;
       15;
       1000000;
       25000];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];

optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(:,1:3), beq, optimal_values(1:3));

%% Optimization with normalization with minmax
vars = {'x1','x2','x3','n1','n2','n3','n4','n5','p1','p2','p3','p4','p5', 'd'};

obj_fun = create_equation(vars, { 'd' }, ...
                                [ 1 ]);
                            
small_dev_rest = create_equation(vars, { 'n1', 'd'}, ...
                                       [ 1/5   -1 ]);

medium_dev_rest = create_equation(vars, { 'n2', 'd'}, ...
                                        [ 1/10  -1 ]);

large_dev_rest = create_equation(vars, { 'n3', 'd'}, ...
                                       [ 1/15  -1 ]);

budget_dev_rest = create_equation(vars, {    'p4',   'd'}, ...
                                        [ 1/1000000  -1 ]);

space_dev_rest = create_equation(vars, {  'n5',     'p5',  'd'}, ...
                                       [ 1/25000  1/25000  -1 ]);

small_rest = create_equation(vars, {'x1', 'n1', 'p1'}, ...
                                   [ 1     1     -1 ]);

medium_rest = create_equation(vars, {'x2', 'n2', 'p2'}, ...
                                    [ 1     1     -1 ]);

large_rest = create_equation(vars, {'x3', 'n3', 'p3'}, ...
                                   [ 1     1     -1 ]);

budget_rest = create_equation(vars, { 'x1',  'x2',   'x3', 'n4', 'p4'}, ...
                                    [ 18000  33000  45150   1     -1 ]);

space_rest = create_equation(vars, { 'x1',  'x2', 'x3', 'n5', 'p5'}, ...
                                   [ 400    750   1050   1     -1 ]);

intcon = [1 2 3];

A = [ small_dev_rest;
      medium_dev_rest;
      large_dev_rest;
      budget_dev_rest;
      space_dev_rest ];

b = [0;
     0;
     0;
     0;
     0];

Aeq = [ small_rest;
        medium_rest;
        large_rest;
        budget_rest;
        space_rest ];

beq = [5;
       10;
       15;
       1000000;
       25000];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];

optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(:,1:3), beq, optimal_values(1:3));

%% Optimization with normalization and lexicographic 
vars = {'x1','x2','x3','n1','n2','n3','n4','n5','p1','p2','p3','p4','p5'};

small_rest = create_equation(vars, {'x1', 'n1', 'p1'}, ...
                                   [ 1     1     -1 ]);

medium_rest = create_equation(vars, {'x2', 'n2', 'p2'}, ...
                                    [ 1     1     -1 ]);

large_rest = create_equation(vars, {'x3', 'n3', 'p3'}, ...
                                   [ 1     1     -1 ]);

budget_rest = create_equation(vars, { 'x1',  'x2',   'x3', 'n4', 'p4'}, ...
                                    [ 18000  33000  45150   1     -1 ]);

space_rest = create_equation(vars, { 'x1',  'x2', 'x3', 'n5', 'p5'}, ...
                                   [ 400    750   1050   1     -1 ]);

intcon = [1 2 3];

A = [];
b = [];

Aeq = [ small_rest;
        medium_rest;
        large_rest;
        budget_rest;
        space_rest];

beq = [5;
       10;
       15;
       1000000;
       25000];
   
lb = [0 0 0 0 0 0 0 0 0 0 0 0 0];

obj_fun = create_equation(vars, { 'n1'}, ...
                                [ 1/5 ]);

fprintf('\n!!!!!!!!!!!!!First execution!!!!!!!!!!!!!\n\n')
optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(:,1:3), beq, optimal_values(1:3));

obj_fun = create_equation(vars, { 'n1', 'n2' }, ...
                                [ 1/5   1/10 ]);
                            
first_execution = create_equation(vars, { 'n1' }, ...
                                        [  1 ]);
                                    
Aeq = [ Aeq;
        first_execution ];
    
beq = [ beq;
        0 ];
    
fprintf('\n!!!!!!!!!!!!!Second execution!!!!!!!!!!!!!\n\n')
optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(1:5,1:3), beq, optimal_values(1:3));

obj_fun = create_equation(vars, { 'n1', 'n2' 'n3' }, ...
                                [ 1/5   1/10 1/15 ]);

second_execution = create_equation(vars, { 'n2' }, ...
                                         [  1 ]);

Aeq = [ Aeq;
        second_execution ];
    
beq = [ beq;
        0 ];

fprintf('\n!!!!!!!!!!!!!Third execution!!!!!!!!!!!!!\n\n')
optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(1:5,1:3), beq, optimal_values(1:3));

obj_fun = create_equation(vars, { 'n1', 'n2' 'n3'     'p4'   }, ...
                                [ 1/5   1/10 1/15  1/1000000 ]);

third_execution = create_equation(vars, { 'n3' }, ...
                                         [  1 ]);

Aeq = [ Aeq;
        third_execution ];
    
beq = [ beq;
        0 ];

fprintf('\n!!!!!!!!!!!!!Fourth execution!!!!!!!!!!!!!\n\n')
optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(1:5,1:3), beq, optimal_values(1:3));

obj_fun = create_equation(vars, { 'n1', 'n2', 'n3',   'p4',      'n5',    'p5'}, ...
                                [ 1/5   1/10  1/15  1/1000000  1/25000  1/25000 ]);

fourth_execution = create_equation(vars, { 'p4' }, ...
                                         [  1 ]);

Aeq = [ Aeq;
        fourth_execution ];
    
beq = [ beq;
        97250 ];

fprintf('\n!!!!!!!!!!!!!Fifth execution!!!!!!!!!!!!!\n\n')
optimal_values = intlinprog(obj_fun,intcon,A,b,Aeq,beq,lb);
display_answer(vars, optimal_values);
display_goals(Aeq(1:5,1:3), beq, optimal_values(1:3));