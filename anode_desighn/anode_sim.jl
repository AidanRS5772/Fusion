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
    b_radius = maximum([norm(v.position) for v in verts])
    new_vertices = StructVector{typeof(verts[1])}((
        position=[Point{3,Float32}(v.position .* scale_factor ./ b_radius) for v in verts],
        normals=verts.normals
    ))

    return GeometryBasics.Mesh(new_vertices, faces(mesh))
end

function sample_triangle(v1, v2, v3, N)
    points = [v1, v2, v3]
    e1 = v2 .- v1
    e2 = v3 .- v1
    r1 = rand(N)
    r2 = rand(N)

    for i in 1:N
        u = r1[i] * (1 - r2[i])
        v = r2[i] * (1 - r1[i])
        push!(points, e1 .* u .+ e2 .* v .+ v1)
    end

    return points
end

function build_kd_tree(mesh, N)
    bb_vertices = [0, 0, 0]
    for i in eachindex(mesh)
        v1 = mesh[i][1].position
        v2 = mesh[i][2].position
        v3 = mesh[i][3].position
        bb_vertices = hcat(bb_vertices, sample_triangle(v1, v2, v3, N)...)
    end

    return KDTree(bb_vertices[:, 2:end])
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

function find_path(mesh, kd_tree, N_kd, sphere_radius, max_orbit, E, u0)
    orbit_cnt = 0
    p0 = u0[1:3]
    p1 = u0[1:3]
    n0 = norm(p1)
    n1 = n0
    function condition(u, t, integrator)
        p = u[1:3]
        n = norm(u[1:3])
        if n0 < n1 && n < n1 && sphere_radius < n1
            orbit_cnt += 1
            if orbit_cnt >= max_orbit
                return true
            end
        end

        idxs = inrange(kd_tree, p, norm(p1 .- p))
        idxs .-= 1
        idxs .÷= 3 + N_kd
        idxs .+= 1
        unique!(idxs)

        @inbounds for j in idxs
            v1 = SVector{3}(mesh[j][1].position)
            v2 = SVector{3}(mesh[j][2].position)
            v3 = SVector{3}(mesh[j][3].position)
            if intersect(p, p1, v1, v2, v3)
                return true
            end
        end

        p0, p1, n0, n1 = p1, p, n1, n

        return false
    end

    cb = DiscreteCallback(condition, terminate!)
    prob = ODEProblem(lorenz!, u0, (0.0, 1.0), (E))
    sol = solve(prob, Rodas5P(autodiff=false), callback=cb)

    return orbit_cnt, sol
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
    E = p
    du[1:3] = u[4:6]
    du[4:6] = e_m .* E(u[1:3])
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
    kd_tree = build_kd_tree(mesh, N_kd)
    c, a = centroid_areas(mesh)
    E_feild(ρ) = E(ρ, c, a, r, R, V)

    println("Start Simulation for $(app_cnt) Appratures")
    initials = Vector{MVector{6}}(undef, N_samples)
    orbit_cnts = Vector{Int}(undef, N_samples)
    uni = rand(6 * N_samples)

    @inbounds for i in 1:N_samples
        u0 = make_point(uni[6*i-5:6*i], r, R, T)
        initials[i] = u0
        orbit_cnts[i], _ = find_path(mesh, kd_tree, N_kd, r, max_orbit, E_feild, u0)
    end

    pdf_trace = histogram(
        x=orbit_cnts,
        opacity=0.5,
        nbinsx=max_orbit,
        histnorm="probability",
        marker_color="rgb(100, 150, 200)"
    )

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

    fig = plot(pdf_trace)
    savefig(fig, "anode_data_plots/appratures_$(app_cnt).html")
    run(`open anode_data_plots/appratures_$(app_cnt).html`)
end

function do_analysis_on_chunk(r, R, V, T, N_samples, N_kd, max_orbit, chunk)
    for app_cnt in chunk
        do_analysis(r, R, V, T, N_kd, N_samples, max_orbit, app_cnt)
    end
end

r, R, V, T = 0.05, 0.5, 1e5, 300
N_samples = 10000
N_kd = 4
max_orbit = 50

nt = nthreads()
println("$(nt) Threads Running...")
chunks = [(6+2*i):2*nt:362 for i in 0:(nt-1)]
for chunk in chunks
    @spawn do_analysis_on_chunk(r, R, V, T, N_samples, N_kd, max_orbit, chunk)
end
