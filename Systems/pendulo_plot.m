subplot(3,1,1)
plot(tout, u_simulink, 'g', "linewidth", 2)
title("Entrada vs Tiempo")
xlabel("Tiempo (s)")
ylabel("Fuerza (N)")
legend("u")
grid on

subplot(3,1,2)
plot(tout, x1_simulink, 'b', "linewidth", 2)
title("Posicion vs Tiempo")
xlabel("Tiempo (s)")
ylabel("Posición (m)")
legend("x1")
grid on

subplot(3,1,3)
plot(tout, x3_simulink, 'r', "linewidth", 2)
title("Ángulo vs Tiempo")
xlabel("Tiempo (s)")
ylabel("Ángulo (Rad)")
legend("x3")
grid on