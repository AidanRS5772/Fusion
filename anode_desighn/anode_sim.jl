using MeshIO
using GeometryBasics
using FileIO
using LinearAlgebra
using DifferentialEquations
using StructArrays
using BenchmarkTools
using .Threads
using StaticArrays
using PlotlyJS
using Random

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
        base = (idx-1)*3  # each triangle starts 3 vertices after the previous
        push!(i, base)
        push!(j, base+1)
        push!(k, base+2)
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

    result = Vector{Int}(undef, (x_max-x_min+1)*(y_max-y_min+1)*(z_max-z_min+1))
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

function find_path(mesh, prox_map, sphere_radius, max_pass, E, B, ρ0, v0)
    function collision_condition(u, t, integrator)
        p1 = @view u[1:3]
        if length(integrator.sol) > 1
            p2 = @view integrator.sol[end-1][1:3]

            valid_idxs = unique!(vcat(prox_map.cells[bounding_box_idxs(p1, p2, bound=prox_map.bound, N=prox_map.N)]...))
            @inbounds for j in valid_idxs
                v1 = SVector{3}(mesh[j][1].position)
                v2 = SVector{3}(mesh[j][2].position)
                v3 = SVector{3}(mesh[j][3].position)
                if intersect(p2, p1, v1, v2, v3)
                    return true
                end
            end
        end
        return false
    end

    pass = 0
    last_r = 0.0
    function sphere_condition(u, t, integrator)
        r = norm(u[1:3])
        if last_r != 0  # skip first point
            if (last_r < sphere_radius && r >= sphere_radius) || (last_r > sphere_radius && r <= sphere_radius)
                pass += 1
                if pass >= max_pass
                    return true
                end
            end
        end
        last_r = r
        return false
    end

    cb1 = DiscreteCallback(collision_condition, terminate!)
    cb2 = DiscreteCallback(sphere_condition, terminate!)
    cb = CallbackSet(cb1, cb2)

    u0 = MVector{6,Float64}([ρ0..., v0...])
    prob = ODEProblem(lorenz!, u0, (0.0, Inf), (E, B))
    sol = hcat(solve(prob, Tsit5(), callback=cb, abstol = 1e-9, adaptive=true).u...)

    return sol[1:3, :], sol[4:6, :]
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

r, R, V = 0.05, 0.25, 1e5
mesh = scale_mesh(load("anode_meshes/appratures_14.stl"), r)
N_cell = 20
prox_map_intersection = make_prox_map_intersection(mesh, N_cell)

c, a = centroid_areas(mesh)
E_feild(ρ) = E(ρ, c, a, r, R, V)
B_feild(ρ) = SVector{3,Float64}(0.0, 0.0, 0.0)

ρ0 = [0, 0.00, 0.07]
v0 = [0, 0, 0]
max_pass = 100
path, vel = find_path(mesh, prox_map_intersection, r, max_pass, E_feild, B_feild, ρ0, v0)

traces = plot_path(mesh, path)
layout = Layout(
    scene=attr(
        aspectmode="cube",  # This forces cubic aspect ratio
        aspectratio=attr(x=1, y=1, z=1),  # Equal scaling on all axes
        xaxis=attr(range=[-R, R]),
        yaxis=attr(range=[-R, R]),
        zaxis=attr(range=[-R, R])
    )
)

plot(traces, layout)
