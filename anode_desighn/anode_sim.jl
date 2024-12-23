using MeshIO
using GeometryBasics
using FileIO
using LinearAlgebra
using DifferentialEquations
using StructArrays
using StaticArrays
using PlotlyJS
using Random
using SpecialFunctions
using Distributions
using Statistics
using JSON

function plot_path(mesh, path)
    # For each triangle, get its three vertices
    x = Float64[]
    y = Float64[]
    z = Float64[]
    i = Int[]
    j = Int[]
    k = Int[]

    # For each triangle in the mesh
    for (idx, triangle) in enumerate(mesh)
        # Get the three vertices
        v1, v2, v3 = triangle[1], triangle[2], triangle[3]

        # Add vertex coordinates
        push!(x, v1[1], v2[1], v3[1])
        push!(y, v1[2], v2[2], v3[2])
        push!(z, v1[3], v2[3], v3[3])

        # Add face indices (3 vertices per triangle)
        base = (idx - 1) * 3  # each triangle starts 3 vertices after the previous
        push!(i, base)
        push!(j, base + 1)
        push!(k, base + 2)
    end

    # Create mesh trace
    mesh_trace = mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=0.5,
        color="grey"
    )

    # Create path trace
    path_trace = scatter3d(
        x=path[1, :],
        y=path[2, :],
        z=path[3, :],
        mode="lines",
        line=attr(
            color="red",
            width=2
        )
    )

    return [mesh_trace, path_trace]
end

function scale_mesh(mesh, scale_factor)
    verts = coordinates(mesh)

    new_vertices = StructVector{typeof(verts[1])}((
        position=[Point{3,Float32}(v.position .* scale_factor) for v in verts],
        normals=verts.normals
    ))

    return GeometryBasics.Mesh(new_vertices, faces(mesh))
end

function bounding_box_idxs(points...; bound, N)
    if all([any(abs.(p) .> bound) for p in points])
        return Int[]
    end

    len = 2 * bound / N
    x_min, x_max = clamp.(floor.(Int, (extrema(p[1] for p in points) .+ bound) ./ len), 0, N - 1)
    y_min, y_max = clamp.(floor.(Int, (extrema(p[2] for p in points) .+ bound) ./ len), 0, N - 1)
    z_min, z_max = clamp.(floor.(Int, (extrema(p[3] for p in points) .+ bound) ./ len), 0, N - 1)

    result = Vector{Int}(undef, (x_max - x_min + 1) * (y_max - y_min + 1) * (z_max - z_min + 1))
    idx = 1
    for z in z_min:z_max, y in y_min:y_max, x in x_min:x_max
        result[idx] = 1 + x + y * N + z * N^2
        idx += 1
    end
    return result
end

struct ProxMap
    cells::Vector{Vector{Int}}
    bound::Float64
    N::Int
end

function make_prox_map_intersection(mesh, N)
    b = maximum([p for point in coordinates(mesh) for p in point])
    cell_sets = [Set() for _ in 1:N^3]
    for i in eachindex(mesh)
        v1 = Vector(mesh[i][1])
        v2 = Vector(mesh[i][2])
        v3 = Vector(mesh[i][3])
        for idx in bounding_box_idxs(v1, v2, v3, bound=b, N=N)
            push!(cell_sets[idx], i)
        end
    end
    cells = [sort(collect(S)) for S in cell_sets]

    return ProxMap(cells, b, N)
end

function intersect(p1, p2, v1, v2, v3)
    d = p2 .- p1
    e1 = v2 .- v1
    e2 = v3 .- v1
    n = cross(e1, e2)
    if abs(d ⋅ n) / (norm(d) * norm(n)) < 1e-6
        return false
    end

    M = @SMatrix [-d[1] e1[1] e2[1]
        -d[2] e1[2] e2[2]
        -d[3] e1[3] e2[3]]
    b = SVector{3}(p1 .- v1)
    t, u, v = M \ b

    if (0.0 <= t <= 1.0) && (0.0 <= u) && (0.0 <= v) && (u + v <= 1.0)
        return true
    end

    return false
end

function find_path(mesh, prox_map, sphere_radius, max_pass, E, B, u0)
    pass_cnt = 0
    p2 = u0[1:3]
    r2 = norm(p2)
    function condition(u, t, integrator)
        p1 = u[1:3]
        r1 = norm(p1)
        if (r1 <= sphere_radius <= r2) || (r2 <= sphere_radius <= r1)
            pass_cnt += 1
            if pass_cnt >= max_pass
                return true
            end
        end

        valid_idxs = unique!(vcat(prox_map.cells[bounding_box_idxs(p1, p2, bound=prox_map.bound, N=prox_map.N)]...))
        @inbounds for j in valid_idxs
            v1 = SVector{3}(mesh[j][1].position)
            v2 = SVector{3}(mesh[j][2].position)
            v3 = SVector{3}(mesh[j][3].position)
            if intersect(p2, p1, v1, v2, v3)
                return true
            end
        end

        p2 = p1
        r2 = r1
        return false
    end

    cb = DiscreteCallback(condition, terminate!)

    prob = ODEProblem(lorenz!, u0, (0.0, 1.0), (E, B))
    sol = solve(prob, Rodas5P(autodiff=false), callback=cb)

    return pass_cnt, sol
end

function centroid_areas(mesh)
    centroids = []
    areas = []

    for i in eachindex(mesh)
        v1 = SVector{3,Float64}(mesh[i][1])
        v2 = SVector{3,Float64}(mesh[i][2])
        v3 = SVector{3,Float64}(mesh[i][3])
        push!(centroids, (v1 .+ v2 .+ v3) ./ 3)

        e1 = v2 .- v1
        e2 = v3 .- v1
        push!(areas, norm(cross(e1, e2)) / 2)
    end

    return centroids, areas ./ sum(areas)
end

@fastmath function E(p, c, a, r, R, V)
    E_tot = @MVector zeros(3)
    @inbounds @simd for i in eachindex(c)
        diff = p .- c[i]
        E_tot .+= (a[i] .* diff) ./ norm(diff)^3
    end
    return E_tot .* (-V / (1 / r - 1 / R))
end

@fastmath function lorenz!(du, u, p, t)
    e_m = 47917944.84
    E, B = p
    r = u[1:3]
    dr = u[4:6]
    du[1:3] = dr
    du[4:6] = e_m .* (E(r) .+ cross(dr, B(r)))
end

function make_point(u, r, R, T)
    ρ = (u[1] * (R^3 - r^3) + r^3)^(1 / 3)
    θ = 2 * π * u[2]
    sϕ = 2 * sqrt(1 - u[3]) * sqrt(u[3])
    v = (0.01100397172 / sqrt(T)) .* erfinv.(2 .* u[4:6] .- 1)

    return MVector{6}(
        ρ * cos(θ) * sϕ,
        ρ * sin(θ) * sϕ,
        ρ * (1 - 2 * u[3]),
        v[1],
        v[2],
        v[3]
    )
end

function do_analysis(r, R, V, T, app_cnt, N_cell, N_samples, max_orbit)
    println("Analysis For $(app_cnt) Appratures:")
    mesh = scale_mesh(load("anode_meshes/appratures_$(app_cnt).stl"), r)
    prox_map_intersection = make_prox_map_intersection(mesh, N_cell)
    c, a = centroid_areas(mesh)
    E_feild(ρ) = E(ρ, c, a, r, R, V)
    B_feild(_) = SVector{3,Float64}(0.0, 0.0, 0.0)

    initials = Vector{MVector{6}}(undef, N_samples)
    pass_cnts = Vector{Int}(undef, N_samples)
    uni = rand(6 * N_samples)
    print("0%")
    @inbounds for i in 1:N_samples
        u0 = make_point(uni[6*i-5:6*i], r, R, T)
        initials[i] = u0
        pass_cnts[i], _ = find_path(mesh, prox_map_intersection, r, 2 * max_orbit + 1, E_feild, B_feild, u0)
        print("\r$(round(i*100/N_samples, digits = 2))%")
    end

    full_orbits = []
    partial_orbit_cnt = 0
    eternal_initials = []
    for i in 1:N_samples
        orbit_cnt = pass_cnts[i] ÷ 2
        if orbit_cnt != max_orbit
            push!(full_orbits, orbit_cnt)
        else
            push!(eternal_initials, initials[i])
        end
        if pass_cnts[i] % 2 == 0
            partial_orbit_cnt += 1
        end
    end

    pdf_trace = histogram(
        x=full_orbits,
        opacity=0.5,
        nbinsx=max_orbit,
        histnorm="probability",
        marker_color="rgb(100, 150, 200)"
    )

    no_pass_rate = partial_orbit_cnt / sum(full_orbits)
    println("\nNo Pass Rate: ", no_pass_rate)
    mean_orbit = mean(full_orbits)
    println("Mean Orbit Count: ", mean_orbit)

    full_orbits .+= 1
    α, β = Distributions.params(fit(Gamma, Float64.(full_orbits)))
    #println("α: $(α)")
    #println("β: $(β)")

    P_expected_tail = 1 - cdf(Gamma(α, β), max_orbit + 1)
    P_observed_tail = length(eternal_initials) / N_samples
    eternal_prop = P_observed_tail - P_expected_tail
    #println("Proportionl of Phase Space with an Eternal Path: ", eternal_prop)

    file = "anode_data/appratures_$(app_cnt).json"
    data = JSON.parsefile(file)
    data["mean_orbit"] = mean_orbit
    data["α"] = α
    data["β"] = β
    data["eternal_prop"] = eternal_prop
    data["eternal_initials"] = eternal_initials
    data["no_pass_rate"] = no_pass_rate

    open(file, "w") do f
        JSON.print(f, data)
    end

    fig = plot(pdf_trace)
    savefig(fig, "anode_data_plots/appratures_$(app_cnt).html")
    run(`open anode_data_plots/appratures_$(app_cnt).html`)
end

r, R, V, T = 0.05, 0.5, 1e5, 300
N_cell = 20
N_samples = 100
max_orbit = 50
app_cnt = 6


mesh = scale_mesh(load("anode_meshes/appratures_$(app_cnt).stl"), r)
prox_map_intersection = make_prox_map_intersection(mesh, N_cell)
c, a = centroid_areas(mesh)
E_feild(ρ) = E(ρ, c, a, r, R, V)
B_feild(_) = SVector{3,Float64}(0.0, 0.0, 0.0)

u0 = make_point(rand(6), r, R, T)
pass_cnt, sol = find_path(mesh, prox_map_intersection, r, 2 * max_orbit, E_feild, B_feild, u0)

println(pass_cnt)

sol = hcat(sol.u...)
plot(plot_path(mesh, scale))
