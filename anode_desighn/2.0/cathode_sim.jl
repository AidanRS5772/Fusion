using Base: Enumerate
using JSON3
using Gmsh
using LinearAlgebra

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

    return abstract_edges, connectivity
end

function find_loops!(connectivity, visited, start_vertex, vertex, path, all_loops)
    visited[vertex] = true
    push!(path, vertex)
    for connection in connectivity[vertex]
        if connection == start_vertex && length(path) > 2
            push!(all_loops, copy(path))
        elseif !visited[connection]
            find_loops!(connectivity, visited, start_vertex, connection, path, all_loops)
        end
    end
    visited[vertex] = false
    pop!(path)
end

function is_loop_facial(loop, connectivity)
    visited = falses(length(connectivity))
    start = rand(1:length(connectivity))
    while start in loop
        start = rand(1:length(connectivity))
    end

    visited = falses(length(connectivity))
    visited[loop] .= true
    queue = [start]
    while !isempty(queue)
        vertex = popfirst!(queue)
        visited[vertex] = true
        for connection in connectivity[vertex]
            if !(connection in loop) && !visited[connection]
                push!(queue, connection)
            end
        end

        if all(visited)
            return true
        end
    end

    return false
end

function find_faces(connectivity)
    all_loops = []
    visited = falses(length(connectivity))

    for vertex in eachindex(connectivity)
        find_loops!(connectivity, visited, vertex, vertex, [], all_loops)
    end

    no_chord_loops = []
    for loop in all_loops
        chord = false
        for i in 0:(length(loop)-1)
            vertex = loop[i+1]
            next_vertex = loop[mod((i + 1), length(loop))+1]
            prev_vertex = loop[mod((i - 1), length(loop))+1]
            for connection in connectivity[vertex]
                if connection != next_vertex && connection != prev_vertex
                    if connection in loop
                        chord = true
                        break
                    end
                end
            end
            if chord
                break
            end
        end
        if !chord
            push!(no_chord_loops, loop)
        end
    end

    facial_loops = []
    for loop in no_chord_loops
        if is_loop_facial(loop, connectivity)
            push!(facial_loops, loop)
        end
    end

    unique!(x -> sort(x), facial_loops)
    return facial_loops
end

app_cnt = 12
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_geometry/app_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
abstract_edges, connectivity = conectivity_analysis(vertices, edges)
abstract_faces = find_faces(connectivity)


display(vertices)
display(abstract_edges)
display(connectivity)
display(abstract_faces)

# scaled_anode_radius = anode_radius / cathode_radius
# scaled_wire_radius = wire_radius / cathode_radius

# try
#     gmsh.initialize()
#     model = gmsh.model
#     model.add("Fusion")
#     mesh_size = scaled_wire_radius / 8

#     geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
#     geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]

#     model.geo.synchronize()
#     model.mesh.generate(1)
#     gmsh.fltk.run()
# catch e
#     println("Error: ", e)
#     stacktrace()
# finally
#     gmsh.finalize()
# end
