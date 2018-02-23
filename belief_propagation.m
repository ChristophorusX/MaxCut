n = 20;
q = 2;
aux = rand(n, n);
E = (aux + aux')/2<0.1;
for i = 1:n / 2
    for j = i + 1:n / 2
        E(i, j) = rand() < 0.8;
        E(j, i) = E(i, j);
        E(n / 2 + i, n / 2 + j) = rand() < 0.8;
        E(n / 2 + j, n / 2 + i) = E(n / 2 + i, n / 2 + j);
    end
end
phi = rand(n, n, q) * 2;
phi_new = zeros(n, n, q);
pin = 0.8;
pout = 0.2;
p = [pin, pout; pout, pin];

for iter = 1:10

    for i = 1:n
        for j = 1:n
            for r = 1:q
                auxprod = 1;
                for k = 1:n
                    if and(E(i, k) == 1, k ~= j)
                        auxsum = phi(k, i, 1) * p(r, 1) + phi(k, i, 2) * p(r, 2);
                        auxprod = auxprod * auxsum;
                    end
                end
                phi_new(i, j, r) = auxprod;
            end
            aux = phi_new(i, j, 1) + phi_new(i, j, 2);
            phi_new(i, j, 1) = phi_new(i, j, 1) / aux;
            phi_new(i, j, 2) = phi_new(i, j, 2) / aux;
        end
    end

    phi = phi_new;
    phi(:, :, 1)
end

vec = zeros(n, 2);
for i = 1:n
    for r = 1:2
        auxprod = 1;
        for j = 1:n
            if E(i, j) == 1
                auxsum = phi(j, i, 1) * p(r, 1) + phi(j, i, 2) * p(r, 2);
                auxprod = auxprod * auxsum;
            end
        end
        vec(i, r) = auxprod;
    end
end

vec
