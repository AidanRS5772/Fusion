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
using NearestNeighbors
using Base.Threads

function cube_layout(lim)
    return Layout(
        scene=attr(
            aspectmode="cube",
            xaxis=attr(range=[-lim, lim]),
            yaxis=attr(range=[-lim, lim]),
            zaxis=attr(range=[-lim, lim])
        )
    )
end

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

function find_path(mesh, bounds, max_orbit, c, a, u0)
    orbit_cnt = 0
    p1 = u0[1:3]
    n0 = norm(p1)
    n1 = n0
    function condition(v, u, t, integrator)
        p = u
        n = norm(u)
        if n0 < n1 && n < n1 && bounds[2] < n1
            orbit_cnt += 1
            if orbit_cnt >= max_orbit
                return true
            end
        end

        if (min(n1, n) <= bounds[2] || max(n1, n) >= bounds[1])
            @inbounds for j in eachindex(mesh)
                v1 = SVector{3}(mesh[j][1].position)
                v2 = SVector{3}(mesh[j][2].position)
                v3 = SVector{3}(mesh[j][3].position)
                if intersect(p, p1, v1, v2, v3)
                    return true
                end
            end
        end

        p1, n0, n1 = p, n1, n

        return false
    end

    cb = DiscreteCallback(condition, terminate!)
    prob = SecondOrderODEProblem(lorenz!, u0[4:6], u0[1:3], (0.0, 10000.0), (c, a))
    sol = solve(prob, DPRKN6(), callback=cb)

    return orbit_cnt, sol
end

@fastmath function lorenz!(dv, v, u, p, t)
    c, a = p
    dv .= 0
    @inbounds @simd for i in eachindex(c)
        diff = u .- c[i]
        dv .+= (a[i] .* diff) ./ norm(diff)^3
    end
end

function centroid_areas(mesh, r, R, V)
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

    return centroids, areas .* (-47917944.84 * V / ((1 / r - 1 / R) * sum(areas)))
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

function do_analysis(r, R, V, T, N_kd, N_samples, max_orbit, app_cnt)
    mesh = scale_mesh(load("anode_meshes/appratures_$(app_cnt).stl"), r)
    c, a = centroid_areas(mesh)
    kd_tree = KDTree(hcat(Vector.(c)...))
    bounds = extrema([norm(v.position) for v in coordinates(mesh)])
    E_feild(ρ) = E(ρ, c, a, r, R, V)

    println("Start Simulation for $(app_cnt) Appratures")
    initials = Vector{MVector{6}}(undef, N_samples)
    orbit_cnts = Vector{Int}(undef, N_samples)
    uni = rand(6 * N_samples)

    @inbounds for i in 1:N_samples
        u0 = make_point(uni[6*i-5:6*i], r, R, T)
        initials[i] = u0
        orbit_cnts[i], _ = find_path(mesh, kd_tree, N_kd, bounds, max_orbit, E_feild, u0)
    end

    display(orbit_cnts)

    pdf_trace = histogram(
        x=orbit_cnts,
        opacity=0.5,
        nbinsx=max_orbit,
        histnorm="probability",
        marker_color="rgb(100, 150, 200)"
    )

    fig = plot(pdf_trace)
    savefig(fig, "anode_data_plots/appratures_$(app_cnt).html")
    run(`open anode_data_plots/appratures_$(app_cnt).html`)

    eternal_initials = []
    del_idxs = []
    for i in eachindex(orbit_cnts)
        if orbit_cnts[i] == max_orbit
            push!(del_idxs, i)
            push!(eternal_initials, initials[i])
        end
    end
    deleteat!(orbit_cnts, del_idxs)

    println("Analysis For $(app_cnt) Appratures:")
    mean_orbit = mean(orbit_cnts)
    println("Mean Orbit Count: ", mean_orbit)

    α, β = Distributions.params(fit(Gamma, Float64.(orbit_cnts)))
    println("α: $(α)")
    println("β: $(β)")

    P_expected_tail = 1 - cdf(Gamma(α, β), max_orbit)
    P_observed_tail = length(eternal_initials) / N_samples
    eternal_prop = P_observed_tail - P_expected_tail
    println("Proportionl of Phase Space with an Eternal Path: ", eternal_prop)

    file = "anode_data/appratures_$(app_cnt).json"
    data = JSON.parsefile(file)
    data["mean_orbit"] = mean_orbit
    data["α"] = α
    data["β"] = β
    data["eternal_prop"] = eternal_prop
    data["eternal_initials"] = eternal_initials

    open(file, "w") do f
        JSON.print(f, data)
    end
end

r, R, V, T = 0.05, 0.2, 1e5, 300
N_samples = 100
N_kd = 20
max_orbit = 50
app_cnt = 6

mesh = scale_mesh(load("anode_meshes/appratures_$(app_cnt).stl"), r)
c, a = centroid_areas(mesh, r, R, V)
bounds = extrema([norm(v.position) for v in coordinates(mesh)])

u0 = make_point(rand(6), r, R, T)
@time orbit_cnt, sol = find_path(mesh, bounds, max_orbit, c, a, u0)

sol = hcat(Vector.(sol.u)...)
plot(plot_path(mesh, sol), cube_layout(R))
