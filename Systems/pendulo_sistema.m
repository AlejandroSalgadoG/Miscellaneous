%{
    metodo pendulo_sistema
        u: entrada
        mc : masa carro
        mp : masa pendulo
        l : longitud barra
        g : gravedad
        fc: friccion carro
        fp: friccion pendulo
        x: vector variables estado
            x1: posicion
            x2: velocidad
            x3: angulo
            x4: velocidad angular

    retorna
        dx: vector derivada variable estado

    Ejemplo uso:
         dx = pendulo_sistema(u, mc, mp, l, g, fc, fp, x);
%}

function dx = pendulo_sistema(u, mc, mp, l, g, fc, fp, x)
    dx1 = x(2) ;
    dx2 =  ( mp*l*sin(x(3))*x(4)^2 - mp*g*cos(x(3))*sin(x(3)) - fc*x(2) + u) / ( mc+mp*sin(x(3))^2 ) + ( fp*cos(x(3))*x(4) ) / ( l*( mc+mp*sin(x(3))^2 ) );
    dx3 = x(4);
    dx4 =  ( (mc+mp)*g*sin(x(3)) - cos(x(3))*u + fc*cos(x(3))*x(2) - mp*l*cos(x(3))*sin(x(3))*x(4)^2 ) / ( l*(mc+mp*sin(x(3))^2) ) - ( (mc+mp)*fp*x(4) ) / ( mp*l^2*(mc+mp*sin(x(3))^2) );
    dx = [dx1; dx2; dx3; dx4];
end