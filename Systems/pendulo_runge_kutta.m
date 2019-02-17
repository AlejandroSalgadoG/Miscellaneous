%{
    metodo pendulo_runge_kutta
        u: entrada (funcion del tiempo)
        mc : masa carro
        mp : masa pendulo
        l : longitud barra
        g : gravedad
        fc: friccion carro
        fp: friccion pendulo
        x0: vector inicial
            x0(1): posicion inicial
            x0(2): velocidad inicial
            x0(3): angulo inicial
            x0(4): velocidad angular inicial
        h: paso metodo
        T: tiempo simulacion

    retorna
        x: vector variables estado
            x(1): posicion
            x(2): velocidad
            x(3): angulo
            x(4): velocidad angular
        t: vector tiempo

    Ejemplo de uso:
         [x, t] = pendulo_runge_kutta(@pendulo_u, mc, mp, l, g, fc, fp, x0, h, T);
%}

function [x, t] = pendulo_runge_kutta(u, mc, mp, l, g, fc, fp, x0, h, T)
    t0 = 1; % Primera posicion de un vector es 1    
    
    t = 0:h:T;
    nt = size(t,2); % numero de columnas en t 
    
    x = zeros(4, nt);
   
    x(:,t0) = x0;
    for k = t0:T/h       
        rk1 = pendulo_sistema(u(k), mc, mp, l, g, fc, fp, x(:,k));
        rk2 = pendulo_sistema(u(k), mc, mp, l, g, fc, fp, x(:,k) + rk1*h/2);
        rk3 = pendulo_sistema(u(k), mc, mp, l, g, fc, fp, x(:,k) + rk2*h/2);
        rk4 = pendulo_sistema(u(k), mc, mp, l, g, fc, fp, x(:,k) + rk3*h);
        
        dx = 1/6*(rk1 + 2*rk2 + 2*rk3 + rk4);
        x(:,k+1) = x(:,k) + h*dx;
    end
end