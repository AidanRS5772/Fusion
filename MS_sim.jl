using DifferentialEquations
using PlotlyJS
using Elliptic


function lorentz!(du, u, p, t)
    e_m = 47917944.84
    Ex, Ey, Ez, Bx, By, Bz = p
    x, y, z, dx, dy, dz = u
    du[1] = dx
    du[2] = dy
    du[3] = dz
    du[4] = e_m * (Ex(x, y, z) + dy * Bz(x, y, z) - dz * By(x, y, z))
    du[5] = e_m * (Ey(x, y, z) + dz * Bx(x, y, z) - dx * Bz(x, y, z))
    du[6] = e_m * (Ez(x, y, z) + dx * By(x, y, z) - dy * Bx(x, y, z))
end

r0 = 1.0
offset = 0.1
u0 = [r0, 0.0, 0.0, 0.0, 0.0, 0.0]
r = 0.05
R = 0.5
ω = 0.1
l = 0.05
n = 10
V = 1e5
I = 1e8

null(x, y, z) = 0
Ex(x, y, z) = -(V / (n * π * (1 - r / R))) * (x / (x^2 + y^2))
Ey(x, y, z) = -(V / (n * π * (1 - r / R))) * (y / (x^2 + y^2))
Ez(x, y, z) = -(V / (n * π * (1 - r / R))) * (z / (z^2 + y^2))

μ0 = 1.25663706e-6
Bz_sol(x, y, z) =
    (μ0 * I / (2 * π * sqrt((ω + sqrt(x^2 + y^2))^2 + (l / 2)^2))) *
    (K(4 * ω * sqrt(x^2 + y^2) / ((ω + sqrt(x^2 + y^2))^2 + (l / 2)^2)) + ((ω - sqrt(x^2 + y^2)) / (ω + sqrt(x^2 + y^2))) * Pi(4 * ω * sqrt(x^2 + y^2) / ((ω + sqrt(x^2 + y^2))^2), π / 2, 4 * ω * sqrt(x^2 + y^2) / ((ω + sqrt(x^2 + y^2))^2 + (l / 2)^2)))

Bx_2w(x, y, z) = -(μ0 * I / (2 * π)) * (y / (x^2 + y^2))
By_2w(x, y, z) = (μ0 * I / (2 * π)) * (x / (x^2 + y^2) - z / (z^2 + y^2))
Bz_2w(x, y, z) = (μ0 * I / (2 * π)) * (z / (z^2 + y^2))


tspan = (0.0, 1e-5)
prob = ODEProblem(lorentz!, u0, tspan, (Ex, Ey, null, Bz_sol, null, null))
sol = hcat(solve(prob, RK4, dtmax=1e-10).u...)


plot(
    [
    scatter(
        x=sol[1, :],
        y=sol[2, :],
        mode="line",
    ),
    scatter(
        x=[0.0],
        y=[0.0],
        mode="marker",
    ),
]
)

# plot(
#     scatter(
#         x = sol[1,:],
#         y = sol[2,:],
#         mode = "line"
#     )
# )
