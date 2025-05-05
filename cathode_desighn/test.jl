
kb = 1.380649e-23
T = 293.15
ε0 = 8.854e-12
e = 1.602e-19
ΔV = -4e5
ra = 0.25
rc = 0.05
md = 3.3435837768e-27
U = (2 / 3) * π * ε0 * e * ΔV * (ra * (ra + rc) / (ra^2 + ra * rc + rc^2) - 2 / 3) / (ra / rc - 1)
K = (3 / 2) * kb * T
E = K + U

println("potential: ", U)
println("kenetic: ", K)
println("energy: ", E)

p_scale = sqrt(2 * md * E)
q_scale = E / e

println("p_scale: ", p_scale)
println("q_scale: ", q_scale)
