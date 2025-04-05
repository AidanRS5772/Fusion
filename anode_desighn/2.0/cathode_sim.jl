using JSON3
using Gmsh
using LinearAlgebra

function conectivity_analysis(vertices, edges)
    abstract_edges = []
    for edge in edges
        idx1 = findfirst(x -> x == edge[1], vertices)
        idx2 = findfirst(x -> x == edge[2], vertices)
        push!(abstract_edges, (idx1, idx2))
    end

    conectivity = []
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

        sym_axsis = normalize(vertices[i])
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

        push!(conectivity, Tuple(connection[perm]))
    end

    return abstract_edges, conectivity
end

app_cnt = 14
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.001

json_data = JSON3.read("cathode_geometry/app_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
abstract_edges, conectivity = conectivity_analysis(vertices, edges)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

try
    gmsh.initialize()
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 8

    geo_vertecies = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    geo_edges = [model.geo.addLine(geo_vertecies[i], geo_vertecies[j]) for (i, j) in abstract_edges]

    model.geo.synchronize()
    model.mesh.generate(1)
    gmsh.fltk.run()
catch e
    println("Error: ", e)
    stacktrace()
finally
    gmsh.finalize()
end
