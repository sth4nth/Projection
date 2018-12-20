# min |x0-x|^2
# s.t. A'x+b=0
d = 10
p = 5
x0 = rand(d)
A = rand(d,p)
b = rand(p)

function project(x0, A, b)
    v = (A'*A)\(A'*x0+b)
    x = x0-A*v
end

x = project(x0,A,b)
