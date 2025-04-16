using JSON3
using Gmsh
using LinearAlgebra
using Combinatorics

function conectivity_analysis(vertices, edges)
    abstract_edges = []
    for edge in edges
        idx1 = findfirst(x -> x == edge[1], vertices)
        idx2 = findfirst(x -> x == edge[2], vertices)
        push!(abstract_edges, Tuple([min(idx1, idx2), max(idx1, idx2)]))
    end

    adj_list = []
    for i in eachindex(vertices)
        adj = []
        for edge in abstract_edges
            if edge[1] == i
                push!(adj, edge[2])
            end
            if edge[2] == i
                push!(adj, edge[1])
            end
        end

        sym_axsis = zeros(3)
        for c in adj
            sym_axsis .+= c
        end
        normalize!(sym_axsis)
        b1 = normalize(cross(sym_axsis, vertices[i] - vertices[adj[1]]))
        b2 = normalize(cross(sym_axsis, b1))

        projections = []
        for a in adj
            edge = vertices[i] - vertices[a]
            proj_e = edge - dot(edge, sym_axsis) .* sym_axsis
            push!(projections, (dot(proj_e, b1), dot(proj_e, b2)))
        end

        angles = [atan(y, x) for (x, y) in projections]
        perm = sortperm(angles)

        push!(adj_list, Tuple(adj[perm]))
    end

    return abstract_edges, adj_list
end


app_cnt = 10
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_data/appratures_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
abstract_edges, adj_list = conectivity_analysis(vertices, edges)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

display(vertices)
display(abstract_edges)
display(adj_list)


try
    gmsh.initialize()
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 8

    geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]


    model.geo.synchronize()
    model.mesh.generate(1)
    gmsh.fltk.run()
catch e
    println("Error: ", e)
    stacktrace()
finally
    gmsh.finalize()
end
