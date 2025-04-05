using Base: Enumerate
using JSON3
using Gmsh
using LinearAlgebra

struct Face
    norm::Vector{Float64}
    verticies::Tuple{Int}
end

function conectivity_analysis(vertices, edges)
    abstract_edges = []
    for edge in edges
        idx1 = findfirst(x -> x == edge[1], vertices)
        idx2 = findfirst(x -> x == edge[2], vertices)
        push!(abstract_edges, Tuple([idx1, idx2]))
    end

    connectivity = []
    for i in eachindex(vertices)
        connection = []
        for edge in abstract_edges
            if edge[1] == i
                push!(connection, edge[2])
            end
            if edge[2] == i
                push!(connection, edge[1])
            end
        end

        sym_axsis = zeros(3)
        for c in connection
            sym_axsis .+= c
        end
        normalize!(sym_axsis)
        b1 = normalize(cross(sym_axsis, vertices[i] - vertices[connection[1]]))
        b2 = normalize(cross(sym_axsis, b1))

        projections = []
        for c in connection
            edge = vertices[i] - vertices[c]
            proj_e = edge - dot(edge, sym_axsis) .* sym_axsis
            push!(projections, (dot(proj_e, b1), dot(proj_e, b2)))
        end

        angles = [atan(y, x) for (x, y) in projections]
        perm = sortperm(angles)

        push!(connectivity, Tuple(connection[perm]))
    end

    # norms = []
    # all_vertices = []
    # for (i, connection) in enumerate(connectivity)
    #     vertex = vertices[i]
    #     for c in connection

    #     end
    # end

    return abstract_edges, connectivity
end

struct Ellipse
    center_idx::Int
    cyl_idx::Tuple{Int,Int}
    major::Vector{Float64}
    minor::Vector{Float64}
end

function make_ellipses(center_idx, cyl_idx1, cyl_idx2, radius, vertices)
    vertex = vertices[center_idx]
    e1 = vertices[cyl_idx1] .- vertex
    e2 = vertices[cyl_idx2] .- vertex
    M = normalize(e1) .+ normalize(e2)
    m = cross(e1, e2)
    if dot(m, vertex) < 0
        m .*= -1
    end
    M .*= radius / norm(m)
    m .*= radius / norm(m)
    return Ellipse(center_idx, (cyl_idx1, cyl_idx2), M, m)
end

function make_geo_ellipse(model, mesh_size, geo_vertices, vertices, ellipse::Ellipse)
    c = vertices[ellipse.center_idx]
    p1 = c .+ ellipse.minor
    geo_p1 = model.geo.addPoint(p1[1], p1[2], p1[3], mesh_size)
    p2 = c .+ ellipse.major
    geo_p2 = model.geo.addPoint(p2[1], p2[2], p2[3], mesh_size)

    model.geo.addEllipseArc(geo_p1, geo_vertices[ellipse.center_idx], geo_p2, geo_p2)

    Tuple([
        geo_p1,
        geo_p2
    ])
end

app_cnt = 10
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_geometry/app_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
abstract_edges, connectivity = conectivity_analysis(vertices, edges)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

try
    gmsh.initialize()
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 8

    geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]

    ellipses = []
    for (i, connection) in enumerate(connectivity)
        push!(ellipses, make_ellipses(i, connection[1], connection[end], scaled_wire_radius, vertices))
        for j in 1:(length(connection)-1)
            push!(ellipses, make_ellipses(i, connection[j], connection[j+1], scaled_wire_radius, vertices))
        end
    end

    geo_ellipses = [make_geo_ellipse(model, mesh_size, geo_vertices, vertices, ellipse) for ellipse in ellipses]

    model.geo.synchronize()
    model.mesh.generate(1)
    gmsh.fltk.run()
catch e
    println("Error: ", e)
    stacktrace()
finally
    gmsh.finalize()
end
