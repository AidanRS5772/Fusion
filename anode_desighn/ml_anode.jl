using StaticArrays
using Random
using LinearAlgebra
using PlotlyJS

function potential(points::Vector{MVector{2, Float64}})
	N = length(points)
	U = 0.0
	for i in 1:N-1
		ϕ1, θ1 = points[i]
		for j in i+1:N
			ϕ2, θ2 = points[j]
			d = acos(sin(ϕ1) * sin(ϕ2) + cos(ϕ1) * cos(ϕ2) * cos(θ1 - θ2))
			U += 1 / d + 1 / (π - d) - 4/π
		end
	end

	return U
end

function grad_a(i::Int, points::Vector{MVector{2, Float64}})::MVector{2, Float64}
	ϕ1, θ1 = points[i]
	sϕ1, cϕ1 = sincos(ϕ1)
	dϕ, dθ = 0.0, 0.0
	for j in 1:length(points)
		if i != j
			ϕ2, θ2 = points[j]
			sϕ2, cϕ2 = sincos(ϕ2)
			sθ, cθ = sincos(θ1 - θ2)
			d = acos(sϕ1 * sϕ2 + cϕ1 * cϕ2 * cθ)
			val = (1 / (π - d)^2 - 1 / d^2) / sin(d)
			dϕ += (sϕ1 * cϕ2 * cθ - cϕ1 * sϕ2) * val
			dθ += cϕ2 * sθ * val
		end
	end

	return MVector(dϕ, (cϕ1 / sϕ1) * dθ)
end

function grad_n(i::Int, points::Vector{MVector{2, Float64}}; h = 1e-8)::MVector{2, Float64}
	points[i][1] += h
	pos_ϕ = potential(points)
	points[i][1] -= 2*h
	neg_ϕ = potential(points)
	points[i][1] += h
	dϕ = (pos_ϕ - neg_ϕ)/(2*h)

	points[i][2] += h
	pos_θ = potential(points)
	points[i][2] -= 2*h
	neg_θ = potential(points)
	points[i][2] += h
	dθ = (pos_θ - neg_θ)/(2*h)
	dθ /= sin(points[i][1])

	return MVector(dϕ, dθ)
end

function evolve!(points::Vector{MVector{2, Float64}}; γ = 1e-4, min_grad = 1e-12, max_iter = 10_000_000)
	N = length(points)
	iter = 0
	grad_norm = Inf
	while (iter < max_iter) && (grad_norm > min_grad)
		grad_norm = 0.0
		grad = [grad_a(i, points) for i in 1:N]
		for (p, g) in zip(points, grad)
			grad_norm += g[1]^2 + g[2]^2
			p[1] -= γ * g[1]
			p[2] -= γ * g[2]

			p[1] = mod(p[1], 2 * π)
			p[2] = mod(p[2], 2 * π)
			if p[1] > π
				p[1] = 2 * π - p[1]
				p[2] = mod(p[2] + π, 2 * π)
			end
		end
		iter += 1
	end
end

function optimal(T::Int, N::Int)
	all_points = []
	for _ in 1:T
		points = [MVector(acos(uϕ), 2 * π * uθ) for (uϕ, uθ) in zip(rand(N), rand(N))]
		evolve!(points)
		push!(all_points, points)
	end

	potentials = potential.(all_points)
	min_U, idx = findmin(potentials)
	min_points = all_points[idx]

	return min_points, min_U, min_points
end

function embed_points_R3(points::Vector{MVector{2, Float64}})
	vecs = []
	for p in points
		ϕ, θ = p
		vec = [cos(θ) * sin(ϕ), sin(θ) * sin(ϕ), cos(ϕ)]
		push!(vecs, vec)
		push!(vecs, -1 .* vec)
	end

	z = [0.0, 0.0, 1.0]
	_, idx = findmin([norm(v .- z) for v in vecs])
	v = vecs[idx]
	point_mat = hcat(vecs...)

	K = z * v' - v * z'
	R = I(3) .+ K .+ K^2 ./ (1 + v[3])
	return R * point_mat
end

points, U, all_points = optimal(1, 3)

println("potential: ", U)
display(points)

points_R3 = embed_points_R3(points)

plot(
	scatter3d(
		x = points_R3[1, :],
		y = points_R3[2, :],
		z = points_R3[3, :],
		mode = "markers",
		marker = attr(
			color = "blue",
			size = "3",
		),
	),
	Layout(
		scene = attr(
			aspectratio = attr(x = 1, y = 1, z = 1),
			xaxis = attr(range = [-1.2, 1.2]),
			yaxis = attr(range = [-1.2, 1.2]),
			zaxis = attr(range = [-1.2, 1.2]),
		),
	),
)

