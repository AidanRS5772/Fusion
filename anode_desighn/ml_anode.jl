using StaticArrays
using Random
using LinearAlgebra
using PlotlyJS
using Statistics
using JSON

function potential(points::Vector{MVector{2, Float64}})
	N = length(points)
	U = 0.0
	for i in 1:N-1
		ϕ1, θ1 = points[i]
		sϕ1, cϕ1 = sincos(ϕ1)
		for j in i+1:N
			ϕ2, θ2 = points[j]
			sϕ2, cϕ2 = sincos(ϕ2)
			d = acos(cϕ1 * cϕ2 + sϕ1 * sϕ2 * cos(θ1 - θ2))
			U += 1 / d + 1 / (π - d) - 4 / π
		end
	end

	return U
end

function gradient(points::Vector{MVector{2, Float64}})
	N = length(points)
	scϕ = [sincos(p[1]) for p in points]
	sθ, cθ, val_mat = zeros(N, N), zeros(N, N), zeros(N, N)
	@inbounds @simd for i in 1:N-1
		sϕ1, cϕ1 = scϕ[i]
		_, θ1 = points[i]
		@inbounds @simd for j in i+1:N
			_, θ2 = points[j]
			sθ_val, cθ_val = sincos(θ1 - θ2)
			sϕ2, cϕ2 = scϕ[j]
			d = acos(cϕ1 * cϕ2 + sϕ1 * sϕ2 * cθ_val)
			val = (1 / (π - d)^2 - 1 / d^2) / sin(d)

			val_mat[i, j] = val
			val_mat[j, i] = val
			cθ[i, j] = cθ_val
			cθ[j, i] = cθ_val
			sθ[i, j] = sθ_val
			sθ[j, i] = -sθ_val
		end
	end

	grad = Vector{MVector{2, Float64}}(undef, N)
	@inbounds @simd for i in 1:N
		sϕ1, cϕ1 = scϕ[i]
		dϕ1, dϕ2, dθ = 0.0, 0.0, 0.0
		@inbounds @simd for j in 1:N
			if i != j
				sϕ2, cϕ2 = scϕ[j]
				dϕ1 += cϕ2 * val_mat[i, j]
				dϕ2 += sϕ2 * cθ[i, j] * val_mat[i, j]
				dθ += sϕ2 * sθ[i, j] * val_mat[i, j]
			end
		end
		grad[i] = MVector(sϕ1 * dϕ1 - cϕ1 * dϕ2, dθ)
	end

	return grad
end

function evolve!(points::Vector{MVector{2, Float64}}; γ = 1e-3, min_grad = 1e-3, max_iter = 10_000_000)
	iter = 0
	grad_norm = Inf
	while (iter < max_iter) && (grad_norm > min_grad)
		grad = gradient(points)
		grad_norm = sqrt(sum(norm.(grad) .^ 2))
		for (p, g) in zip(points, grad)
			p[1] -= γ * g[1]
			p[2] -= γ * g[2]

			p[1] = mod(p[1], 2 * π)
			p[2] = mod(p[2], 2 * π)
			if p[1] > π
				p[1] = 2 * π - p[1]
				p[2] = mod(p[2] + π, 2 * π)
			end
		end
	end
end

function optimal(T::Int, N::Int)
	all_points = []
	print("Trials: 0/$(T)")
	for i in 1:T
		points = [MVector(acos(uϕ), 2 * π * uθ) for (uϕ, uθ) in zip(rand(N), rand(N))]
		evolve!(points)
		push!(all_points, points)
		print("\rTrials: $(i)/$(T)")
	end
	print("\rDone")
	println("")

	potentials = potential.(all_points)
	_, idx = findmin(potentials)
	min_points = popat!(all_points, idx)

	return min_points, std(potentials)
end

function embed_points_R3(points::Vector{MVector{2, Float64}})
	vecs = []
	for p in points
		ϕ, θ = p
		vec = [cos(θ) * sin(ϕ), sin(θ) * sin(ϕ), cos(ϕ)]
		push!(vecs, vec)
		push!(vecs, -1 .* vec)
	end

	return hcat(vecs...)
end

function orient!(points_R3::Matrix{Float64})
	z = [0.0, 0.0, 1.0]
	_, idx = findmin([norm(v .- z) for v in eachcol(points_R3)])
	v = points_R3[:, idx]

	K = z * v' - v * z'
	R = I(3) .+ K .+ K^2 ./ (1 + v[3])
	points_R3 .= R * points_R3
end


for i in 45:200
	println("\n$(2*i) Appratures:")
	min_points , std_dev = optimal(64, i)
	points_R3 = embed_points_R3(min_points)
	orient!(points_R3)
	println("Std. Dev. = ", std_dev)
	println("Points: ")
	display(points_R3)
	data = Dict("points" => points_R3, "σ" => std_dev)
	open("anode_data/appratures_$(2*i).json", "w") do io
		JSON.print(io, data)
	end
end



