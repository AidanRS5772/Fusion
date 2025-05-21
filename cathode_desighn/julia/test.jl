using LinearAlgebra

a = normalize([1, 0, 0])
b = normalize([1, 1, 1])
c = normalize([0, 0, 1])

e1 = a - c
e2 = b - c
e3 = a - b
E = [e1 e2]
t = -(E' * E) \ (E' * c)
if (t[1] < 0 || t[2] < 0 || (t[1] + t[2]) > 1)
    norms = []

    t1 = -dot(e1, c) / dot(e1, e1)
    if (0 < t1 < 1)
        push!(norms, norm(e1 * t1 + c))
    end

    t2 = -dot(e2, c) / dot(e2, e2)
    if (0 < t2 < 1)
        push!(norms, norm(e2 * t2 + c))
    end

    t3 = -dot(e3, b) / dot(e3, e3)
    if (0 < t3 < 1)
        push!(norms, norm(e3 * t3 + b))
    end

    if isempty(norms)
        println(1.0)
    else
        println(minimum(norms))
    end
else
    println(norm(E * t + c))
end
