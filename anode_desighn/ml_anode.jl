using LinearAlgebra
using Random
using Statistics
using JSON
using Base.Threads

struct Point
    θ::Float64
    ϕ::Float64
end

dist(p1::Point, p2::Point) = acos(cos(p1.ϕ) * cos(p2.ϕ) + sin(p1.ϕ) * sin(p2.ϕ) * cos(p1.θ - p2.θ))

function potential(p::Vector{Point})
    N = length(p)
    dist_matrix = [dist(p[i], p[j]) for i in 1:N-1 for j in i+1:N]
    U = sum(1 / d + 1 / (π - d) - 4 / π for d in dist_matrix)
    return U
end

function translate_ϕ(points::Vector{Point}, i::Int, dϕ::Float64)
	p = copy(points)
    ϕ = p[i].ϕ + dϕ
    θ = p[i].θ
    if ϕ > π
        θ = mod(θ + π, 2 * π)
        ϕ = 2 * π - ϕ
    elseif ϕ < 0
        θ = mod(θ + π, 2 * π)
        ϕ = -ϕ
    end
    p[i] = Point(θ, ϕ)
	return p
end

function translate_ϕ!(p::Vector{Point}, i::Int, dϕ::Float64)
    ϕ = p[i].ϕ + dϕ
    θ = p[i].θ
    if ϕ > π
        θ = mod(θ + π, 2 * π)
        ϕ = 2 * π - ϕ
    elseif ϕ < 0
        θ = mod(θ + π, 2 * π)
        ϕ = -ϕ
    end
    p[i] = Point(θ, ϕ)
end

function translate_θ(points::Vector{Point}, i::Int, dθ::Float64)
	p = copy(points)
    θ = mod(p[i].θ + dθ, 2 * π)
    ϕ = p[i].ϕ
    p[i] = Point(θ, ϕ)
	return p
end

function translate_θ!(p::Vector{Point}, i::Int, dθ::Float64)
    θ = mod(p[i].θ + dθ, 2 * π)
    ϕ = p[i].ϕ
    p[i] = Point(θ, ϕ)
end

function gradient(points::Vector{Point}, h::Float64)
    N = length(points)
    grad = Vector{Point}(undef, N)  # Preallocate
    p = potential(points)
    for i in 1:N
        dθ = (potential(translate_θ(copy(points), i, h)) - p) / h
        dθ /= sin(points[i].ϕ)^2
        dϕ = (potential(translate_ϕ(copy(points), i, h)) - p) / h
        grad[i] = Point(dθ, dϕ)
    end
    return grad
end

function norm_grad(grad::Vector{Point}, points::Vector{Point})
    norm = sum(sin(p.ϕ)^2 * g.θ^2 + g.ϕ^2 for (g, p) in zip(grad, points))
    return sqrt(norm)
end

function evolve!(points::Vector{Point}, h::Float64, γ::Float64, min_norm::Float64, max_iter::Int)
    grad_norm = Inf64
	iter = 0
    while (min_norm < grad_norm) && (iter < max_iter)
        grad = gradient(points, h)
        for i in eachindex(grad)
            translate_θ!(points, i, -γ * grad[i].θ)
            translate_ϕ!(points, i, -γ * grad[i].ϕ)
        end

		iter += 1
        grad_norm = norm_grad(grad, points)
    end
end

function instatiate(N::Int)
    uθ = rand(N)
    uϕ = rand(N)
    points = Vector{Point}(undef, N)  # Preallocate
    for i in 1:N
        θ = 2 * π * uθ[i]
        ϕ = acos(1 - uϕ[i])
        @assert 0.0 ≤ ϕ ≤ π / 2 "ϕ out of bounds in instatiate: $(ϕ)"
        @assert 0.0 ≤ θ ≤ 2π "θ out of bounds in instatiate: $(θ)"
        points[i] = Point(θ, ϕ)
    end
    return points
end

function optimize(trial_cnt::Int, N::Int; h::Float64 = 1e-12, γ::Float64 = 1e-3, min_norm::Float64 = 1e-3, max_iter::Int = 100_000)
    all_points_potentials = Vector{Tuple{Vector{Point}, Float64}}(undef, trial_cnt)

	print("  Progress: 0/$(trial_cnt)")
    @threads for i in 1:trial_cnt
        points = instatiate(N)
        evolve!(points, h, γ, min_norm, max_iter)
        all_points_potentials[i] = (points, potential(points))
		print("\r  Progress: $(i)/$(trial_cnt)")
    end
	print("\r  Done")

    min_potential, min_index = findmin([x[2] for x in all_points_potentials])
    min_points = all_points_potentials[min_index][1]
    all_potentials = [x[2] for x in all_points_potentials]

    return min_points, min_potential, all_potentials
end

function make_points(points::Vector{Point})
    out = Matrix{Float64}(undef, 3, 2 * length(points))  # Preallocate
    for (i, p) in enumerate(points)
        v = [cos(p.θ) * sin(p.ϕ), sin(p.θ) * sin(p.ϕ), cos(p.ϕ)]
        out[:, 2*i-1] .= v
        out[:, 2*i] .= -v
    end
    return out
end

function orient!(points_3d::Matrix{Float64}; ϵ = 1e-4)
    z = [0.0, 0.0, 1.0]
    min_dif = Inf
    v = nothing
    for p in eachcol(points_3d)
        dif = norm(p .- z)
        if dif < min_dif
            min_dif = dif
            v = p
        end
    end

    K = z * v' - v * z'
    R = I(3) .+ K .+ K * K ./ (1 + v[3])
    points_3d .= R * points_3d
end

for i in 29:200
    println("\n$(2*i) Points:")
    points, _, all_potentials = optimize(64, i)
    points_3d = make_points(points)
    orient!(points_3d)
    σ = std(all_potentials)
    println("Standard Dev. of Potentials: ", σ)
    println("Points:")
    display(points_3d)

    data = Dict("σ" => σ, "points" => points_3d)
    open("anode_data/appratures_$(2*i).json", "w") do io
        JSON.print(io, data)
    end
end

