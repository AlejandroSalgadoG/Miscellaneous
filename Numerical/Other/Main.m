%% Bisection
fun = @(x) x(1)^2;

inter = [-3 7];
delta = 0.01;
e = 0.01;

bisection(fun, inter, delta, e);

%% Coordinates 
fun = @(x) x(1)^2 + x(2)^2;

x_0 = [5 5];
e = 0.1;

coordinates(fun, x_0, e);

%% Outter point method
fun = @(x)(x(1)-2)^2 + (x(2)-1)^2;
res = @(x) x(1) + x(2) + 2;

ro = 5;
mu = 2;
x_0 = [0 -2];
e = 0.0001;

x = outter_point(fun, res, ro, mu, x_0, e);

%% Inner point method
fun = @(x) -5*x(1) + x(1)^2 -8*x(2) + 2*x(2)^2;
res = @(x) 6 - 3*x(1) - 2*x(2);

ro = 5;
mu = 0.5;
x_0 = [ -0.4072 1.0309 ];
e = 0.0001;

x = inner_point(fun, res, ro, mu, x_0, e);