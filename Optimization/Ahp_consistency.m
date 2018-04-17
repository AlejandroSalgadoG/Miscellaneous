% matriz de preferencias
A = [
        [ 1,   0.80, 1.10, 1.50]
        [1.25,  1,   1.20, 1.30]
        [0.91, 0.83,  1,   0.95]
        [0.67, 0.77, 1.05,  1  ]
     ];

n = 4; % numero de elementos
 
sum_inv = sum(A,1).^-1;
norm = repmat(sum_inv,n,1);

N = A.*norm
 
W = sum(N,2)./n % pesos
 
 b = A*W;
 n_max = sum(b)
 
 ci = (n_max - n) / (n-1);
 cr = (1.98*(n-2)) / n;
 
 ci/cr