x0 = [ 0 0 pi/4 0 ];
mc=5; mp=1; l=2; g=9.8; fc=0; fp=0.5; h=0.01; T=20;
[x_euler, t_euler] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x0, h, T);

h=0.3;
[x_runge, t_runge] = pendulo_runge_kutta(@pendulo_u, mc, mp, l, g, fc, fp, x0, h, T);

subplot(2,1,1)
plot(t_euler, x_euler(1,:), 'b', t_runge, x_runge(1,:), 'g', tout, x1_simulink, 'black', "linewidth", 2)
title("Posición vs Tiempo")
xlabel("Tiempo (s)")
ylabel("Posición (m)")
legend("euler", "runge-kutta", "simulink", 'Location','southwest')
grid on

subplot(2,1,2)
plot(t_euler, x_euler(3,:), 'b', t_runge, x_runge(3,:), 'g', tout, x3_simulink, 'black', "linewidth", 2)
title("Ángulo vs Tiempo")
xlabel("Tiempo (s)")
ylabel("Ángulo (rad)")
legend("euler", "runge-kutta", "simulink", 'Location','southwest')
grid on

figure
subplot(2,1,1)
[x_euler_1, t_euler_1] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.1, T);
[x_euler_2, t_euler_2] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.01, T);
[x_euler_3, t_euler_3] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.001, T);

plot(t_euler_1, x_euler_1(1,:), t_euler_2, x_euler_2(1,:), t_euler_3, x_euler_3(1,:), "linewidth", 2)
title("Posición vs Tiempo usando método de Euler")
xlabel("Tiempo (s)")
ylabel("Posición (m)")
legend("h=0.1", "h=0.01", "h=0.001")
grid on

subplot(2,1,2)
[x_runge_1, t_runge_1] = pendulo_runge_kutta(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.3, T);
[x_runge_2, t_runge_2] = pendulo_runge_kutta(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.2, T);
[x_runge_3, t_runge_3] = pendulo_runge_kutta(@pendulo_u, mc, mp, l, g, fc, fp, x0, 0.1, T);

plot(t_runge_1, x_runge_1(1,:), t_runge_2, x_runge_2(1,:), t_runge_3, x_runge_3(1,:), "linewidth", 2)
title("Posición vs Tiempo usando método de Runge-Kutta")
xlabel("Tiempo (s)")
ylabel("Posición (m)")
legend("h=0.3", "h=0.2", "h=0.1")
grid on