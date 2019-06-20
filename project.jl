# min |x0-x|^2
# s.t. A'x+b=0
using LinearAlgebra

d = 5
q = 2
x0 = rand(d)
A = rand(d,q)
b = rand(q)

function project(x, A, b)
    # solve primal and dual equations
    v = (A'*A)\(A'*x+b)
    x = x-A*v
end

function kkt(x, A, b)
    # solve KKT master equation
    d, q = size(A)
    I = Diagonal(ones(d))
    O = zeros(q,q)
    M = [I A; A' O]
    y = [x; -b]
    u = M\y
    v = u[d+1:end]
    x = u[1:d]
end

function elim(x, A, b)
    # eliminating constraints
    W = nullspace(A')'
    m = -A'\b
    z = (W*W')\W*(x-m)
    x = W'*z+m
end

function dca(x, A, b)
    # dual coordinate ascent
    steps = 100
    d, q = size(A)
    v = zeros(q)
    for t = 1:steps
        for k = 1:q
            a = A[:,k]
            v[k] = (a'*a)\(a'*(x-A*v+a*v[k])+b[k])
        end
    end
    x = x-A*v
end

function vonNeumann(x, A, b)
    # von Neumann alternative projection
    steps = 100
    for t = 1:steps
        for k = 1:length(b)
            x = project(x,A[:,k],b[k])
        end
    end
    x
end

function dykstra(x, A, b)
    # Dykstra projection
    steps = 100
    d,n = size(A)
    z = zeros(d,n)
    for t = 1:steps
        for i = 1:n
            x0 = x
            x = project(x0+z[:,i], A[:,i], b[i])
            z[:,i] += x0-x
        end
    end
    x
end

x = project(x0,A,b)
xe = elim(x0,A,b)
xa = vonNeumann(x0,A,b)
xd = dykstra(x0,A,b)
xcd = dca(x0,A,b)
