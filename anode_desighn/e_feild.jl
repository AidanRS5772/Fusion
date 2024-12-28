using PlotlyJS
using SpecialFunctions
using JacobiElliptic

@btime SpecialFunctions.ellipk(0.5)
@btime Elliptic.K(0.5)
@btime JacobiElliptic.K(0.5)

@btime Elliptic.Pi(0.2, π / 2, 0.5)
@btime JacobiElliptic.Pi(sqrt(0.2), sqrt(0.5))

val1 = Elliptic.Pi(0.5, π / 2, 0.5)
val2 = JacobiElliptic.Pi(0.5, 0.5)

println(val1)
println(val2)
