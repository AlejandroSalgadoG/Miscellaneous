%{
    metodo pendulo_euler
        u: entrada (funcion del tiempo)
        mc : masa carro
        mp : masa pendulo
        l : longitud barra
        g : gravedad
        fc: friccion carro
        fp: friccion pendulo
        x10: posicion inicial
        x20: velocidad inicial
        x30: angulo inicial
        x40: velocidad angular inicial
        h: paso metodo
        T: tiempo simulacion

    Ejemplo de uso:
         [x1, x3, t] = pendulo_euler(@pendulo_u, mc, mp, l, g, fc, fp, x10, x20, x30, x40, h, T);
%}

function [x1, x3, t] = pendulo_euler(u, mc, mp, l, g, fc, fp, x10, x20, x30, x40, h, T)
    t0 = 1; % Primera posicion de un vector es 1    
    
    t = 0:h:T;
    x1 = 0:h:T;
    x2 = 0:h:T;
    x3 = 0:h:T;
    x4 = 0:h:T;
    
    x1(t0) = x10;
    x2(t0) = x20;
    x3(t0) = x30;
    x4(t0) = x40;
    
    for k = t0:T/h
        x1(k+1) = x1(k) + h*( x2(k) );
        x2(k+1) = x2(k) + h*( ( mp*l*sin(x3(k))*x4(k)^2 - mp*g*cos(x3(k))*sin(x3(k)) - fc*x2(k) + u(k) ) / ( mc+mp*sin(x3(k))^2 ) + ( fp*cos(x3(k))*x4(k) ) / ( l*( mc+mp*sin(x3(k))^2 ) ) );
        x3(k+1) = x3(k) + h*( x4(k) );
        x4(k+1) = x4(k) + h*( ( (mc+mp)*g*sin(x3(k)) - cos(x3(k))*u(k) + fc*cos(x3(k))*x2(k) - mp*l*cos(x3(k))*sin(x3(k))*x4(k)^2 ) / ( l*(mc+mp*sin(x3(k))^2) ) - ( (mc+mp)*fp*x4(k) ) / ( mp*l^2*(mc+mp*sin(x3(k))^2) ) );
    end
end