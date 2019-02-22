%{
    metodo pendulo_euler
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
         x0 = [ 0 0 pi/4 0 ];
         mc=5; mp=1; l=2; g=9.8; fc=0; fp=0.5; h=0.001; T=20;
         [x, t] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x, h, T);
%}

function [x, t] = pendulo_euler(u, mc, mp, l, g, fc, fp, x0, h, T)
    t0 = 1; % Primera posicion de un vector es 1    
    
    t = 0:h:T;
    nt = size(t,2); % numero de columnas en t 
    
    x = zeros(4, nt);
 
    x(:,t0) = x0;
    for k = t0:T/h
        dx = pendulo_sistema(u(k), mc, mp, l, g, fc, fp, x(:,k));
        x(:, k+1) = x(:,k) + h*dx;
    end
end