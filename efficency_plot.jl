using PlotlyJS
using LazyGrids

D_m = 1.875612928e6 #KeV
α = 0.0072973525643
r_d_sqr = 45.18260384 #mb

a = [5.46385e4, 2.70405e2, -7.9849e-2, 1.541285e-5, -1.166645e-9]
S_Factor(E) = a[1] + a[2] * E + a[3] * E^2 + a[4] * E^3 + a[5] * E^4 #KeV*mb

efficency(T) = (D_m * α^2 / (54 * π * T))^(1 / 6) * (S_Factor((T^2 * D_m * π^2 * α^2 / 4)^(1 / 3)) / (4 * r_d_sqr * T)) * exp(-(3 / 2) * (2 * D_m * π^2 * α^2 / T)^(1 / 3))

Temperature(r, R, V) = V * (2 / (3 * r) - (R + r) / (R^2 + R * r + r^2)) / (1 / r - 1 / R)

function my_max(f, a, b; tol = 1e-16, max_iter = 10000)
	ϕ = (sqrt(5) - 1) / 2

	c = b - ϕ * (b - a)
	d = a + ϕ * (b - a)

	f_c = f(c)
	f_d = f(d)

	iter = 0
	while abs(b - a) > tol && iter < max_iter
		if f_c > f_d
			b, d, f_d = d, c, f_c
			c = b - ϕ * (b - a)
			f_c = f(c)
		else
			a, c, f_c = c, d, f_d
			d = a + ϕ * (b - a)
			f_d = f(d)
		end
		iter += 1
	end

	return (a + b) / 2
end

function my_int(f::Function, a, b; err = 1e-16)
	if a == b
		return 0
	end
	val = 1
	if a > b
		a, b = b, a
		val = -1
	end

	ϵ = 1e-4
	f_4(x) = (f(x + 2 * ϵ) - 4 * f(x + ϵ) + 6 * f(x) - 4 * f(x - ϵ) + f(x - 2 * ϵ)) / ϵ^4
	n::Int = 2 * ceil(Int, ((b - a)^5 * my_max(f_4, a, b, tol = 1e-4) / (2880 * err))^(1 / 4)) + 2

	h = (b - a) / n
	X = collect(LinRange(a, b, n))

	sum = (f(X[1]) + f(X[end])) * h / 3
	flag = true
	for x in X[2:end-1]
		if flag
			sum += 4 * h * f(x) / 3
		else
			sum += 2 * h * f(x) / 3
		end
		flag = !flag
	end

	return val * sum
end

V = LinRange(0, 2000, 1000)
r = 0.05
R = 0.5
E = efficency.(Temperature.(r, R, V)) * 100

e(v) = efficency(Temperature(r, R, v)) * 100

val = my_max(e, 0, 1000)
println("Max Voltage: ", val)
println("Max Efficency: ", e(val))


plot(
	[
		scatter(
			x = V,
			y = E,
			mode = "line",
		),
		scatter(
			x = [100, 200, val],
			y = [e(100), e(200), e(val)],
			mode = "markers+text",
            text=string.(round.([e(100), e(200), e(val)], digits = 3)),
            textposition="top center",
            marker=attr(size=10, color = "green")
		)],
	Layout(
		title = "Proportion of Collisions that Result in Fusion",
		xaxis = attr(
			title = "Voltage of Anode (KV)",
			tickmode = "linear",
			tick0 = 0,
			dtick = 200,
		),
		yaxis = attr(
			title = "Fusion Event Proportion (%)",
			tickmode = "linear",
			tick0 = 0,
			dtick = 0.5,
		),
	),
)


# n = 100  
# r_range = LinRange(0.01, 1, n)
# R_range = LinRange(0.01, 1, n)
# V_range = LinRange(0, 150, n)

# r, R, V = mgrid(r_range, R_range, V_range)

# ξ_T(r,R,V) = if r < R return ξ(T(r,R,V)) end

# print("V = 100, R = .5, r = .05, Eff = ", ξ_T(.05,.5,100))
# eff_vals = ξ_T.(r, R, V)

# eff_0 = .10

# plot(
#     isosurface(
#         x=r[:],
#         y=R[:],
#         z=V[:],
#         value=eff_vals[:],
#         isomin = eff_0,
#         isomax = eff_0,
#         caps=attr(x_show=false, y_show=false)
#     )
# )
