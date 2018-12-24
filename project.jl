# min |x0-x|^2
# s.t. A'x+b=0
using LinearAlgebra
using Plots

d = 5
q = 2
x0 = rand(d)
A = rand(d,q)
b = rand(q)

function project(x, A, b)
    # direct method
    v = (A'*A)\(A'*x+b)
    x = x-A*v
end

function project0(x, A, b)
    # master equation method
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
    # Eliminating equality constraints
    W = nullspace(A')'
    m = -A'\b
    z = (W*W')\W*(x-m)
    x = W'*z+m
end

function dcd(x, A, b)
    # dual coordinate descent
    steps = 100
    d, q = size(A)
    v = rand(q)
    for t = 1:steps
        for k = 1:q
            a = A[:,k]
            v[k] = (a'*a)\(a'*(x-A*v+a*v[k])+b[k])
        end
    end
    x = x-A*v
end

function alternative_project(x, A, b)
    steps = 100
    for t = 1:steps
        for k = 1:length(b)
            x = project(x,A[:,k],b[k])
        end
    end
    x
end

function dykstra(x, A, b)
    steps = 100
    d, q = size(A)
    z = zeros(d,q)
    for i=1:steps
        for k = 1:q
            x0 = x
            x = project(x0+z[:,k], A[:,k], b[k])
            z[:,k] = x0+z[:,k]-x
        end
    end
    x
end

x = project(x0,A,b)
xe = elim(x0,A,b)
xa = alternative_project(x0,A,b)
xd = dykstra(x0,A,b)
xcd = dcd(x0,A,b)
