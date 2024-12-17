using MeshIO
using GeometryBasics
using FileIO
using LinearAlgebra
using DifferentialEquations
using DiffEqPhysics
using GLMakie
using StructArrays
using BenchmarkTools
using .Threads
using StaticArrays

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

    return [1 + x + y * N + z * N^2 for x in x_min:x_max for y in y_min:y_max for z in z_min:z_max]
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

function intersect(p1, p2, tri)
    d = p2 .- p1
    v1, v2, v3 = tri
    e1 = v2 .- v1
    e2 = v3 .- v1
    n = cross(e1, e2)
    if abs(d ⋅ n) / (norm(d) * norm(n)) < 1e-6
        return false
    end

    M = [-1.0 .* d e1 e2]
    t, u, v = M \ (p1 .- v1)

    if (0.0 <= t <= 1.0) && (0.0 <= u) && (0.0 <= v) && (u + v <= 1.0)
        return true
    end

    return false
end

function find_path(mesh, prox_map, sphere_radius, max_pass, E, B, ρ0, v0)
    function collision_condition(u, t, integrator)
        p1 = u[1:3]
        if length(integrator.sol) > 1
            p2 = integrator.sol[end-1][1:3]

            valid_idxs = collect(Set(vcat(prox_map.cells[bounding_box_idxs(p1, p2, bound=prox_map.bound, N=prox_map.N)]...)))
            @inbounds for j in valid_idxs
                tri = (Vector(mesh[j][1].position),
                    Vector(mesh[j][2].position),
                    Vector(mesh[j][3].position))
                if intersect(p2, p1, tri)
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

    u0 = [ρ0..., v0...]
    prob = ODEProblem(lorenz!, u0, (0.0, Inf), (E, B))
    sol = hcat(solve(prob, AutoTsit5(Rosenbrock23()), callback=cb).u...)

    return sol[1:3, :], sol[4:6, :]
end

function centroid_areas(mesh)
    centroids = []
    areas = []

    for i in eachindex(mesh)
        v1 = Vector(mesh[i][1])
        v2 = Vector(mesh[i][2])
        v3 = Vector(mesh[i][3])
        push!(centroids, (v1 .+ v2 .+ v3) ./ 3)

        e1 = v2 .- v1
        e2 = v3 .- v1
        push!(areas, norm(cross(e1, e2)) / 2)
    end

    return centroids, areas ./ sum(areas)
end

function E(p, c, a, r, R, V)
    E_tot = zeros(3)
    @inbounds @simd for i in eachindex(c)
        diff = p .- c[i]
        E_tot .+= (a[i] .* diff) ./ norm(diff)^3
    end
    return E_tot .* (-V / (1 / r - 1 / R))
end

function lorenz!(du, u, p, t)
    e_m = 47917944.84
    E, B = p
    r = u[1:3]
    dr = u[4:6]
    du[1:3] = dr
    du[4:6] = e_m .* (E(r) .+ cross(dr, B(r)))
end

r, R, V = 0.05, 0.25, 1e5
mesh = scale_mesh(load("anode_meshes/appratures_30.stl"), r)
prox_map_intersection = make_prox_map_intersection(mesh, 20)

c, a = centroid_areas(mesh)

E_feild(ρ) = E(ρ, c, a, r, R, V)
B_feild(ρ) = [0.0, 0.0, 0.0]

ρ0 = [0, 0.02, 0.1]
v0 = [0, 0, 0]
t_max = 1e-4
path, vel = find_path(mesh, prox_map_intersection, r, 50, E_feild, B_feild, ρ0, v0)

fig = Figure()
ax = Axis3(fig[1, 1],
    aspect=(1, 1, 1),  # fixed aspect ratio
    limits=((-R, R), (-R, R), (-R, R)))  # fixed range for all axes

mesh!(ax, mesh)
lines!(ax, path[1, :], path[2, :], path[3, :], color=:red, linewidth=2)
display(fig)
