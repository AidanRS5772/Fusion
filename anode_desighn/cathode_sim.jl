using JSON3
using Gmsh
using LinearAlgebra
using Combinatorics

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

function find_loops!(connectivity, vertex, parents, visited, all_loops)
    visited[vertex] = true
    for connection in connectivity[vertex]
        if connection == parents[vertex]
            continue
        end
        if visited[connection]
            path = [vertex]
            current = parents[vertex]
            connection_found = false

            while current != 0
                pushfirst!(path, current)
                if current == connection
                    connection_found = true
                    break
                end
                current = parents[current]
            end

            if connection_found
                loop_start = findfirst(v -> v == connection, path)
                loop = path[loop_start:end]
                push!(all_loops, loop)
            end
        else
            parents[connection] = vertex
            find_loops!(connectivity, connection, parents, visited, all_loops)
        end
    end
end

function add_loops(loops)
    # Convert loops to edge sets
    edge_sets = []
    for loop in loops
        edge_set = Set()
        for i in eachindex(loop)
            u = loop[i]
            v = loop[mod1(i + 1, length(loop))]
            edge = (min(u, v), max(u, v))
            push!(edge_set, edge)
        end
        push!(edge_sets, edge_set)
    end

    # Compute symmetric difference
    new_loop_edge_set = symdiff(edge_sets...)

    # If no edges, return nothing
    if isempty(new_loop_edge_set)
        return nothing
    end

    # Build adjacency list
    adj_list = Dict()
    vertices = Set()

    for (u, v) in new_loop_edge_set
        # Add vertices to set
        push!(vertices, u, v)

        # Build adjacency list
        if !haskey(adj_list, u)
            adj_list[u] = []
        end
        if !haskey(adj_list, v)
            adj_list[v] = []
        end

        push!(adj_list[u], v)
        push!(adj_list[v], u)
    end

    # Quick check: Every vertex must have exactly 2 neighbors for a simple cycle
    for (vertex, neighbors) in adj_list
        if length(neighbors) != 2
            return nothing
        end
    end

    # If we have a disconnected graph, it's not a valid cycle
    if length(vertices) != length(adj_list)
        return nothing
    end

    # Construct the cycle
    start = first(vertices)
    cycle = [start]
    prev = nothing
    current = start

    # Use a greedy approach - always take the unused neighbor
    while length(cycle) < length(vertices)
        next_vertex = (adj_list[current][1] != prev) ? adj_list[current][1] : adj_list[current][2]

        if next_vertex in cycle && next_vertex != start
            return nothing
        end

        if next_vertex == start && length(cycle) < length(vertices)
            return nothing
        end

        if length(cycle) < length(vertices)
            push!(cycle, next_vertex)
        end

        prev = current
        current = next_vertex
    end

    if adj_list[cycle[end]][1] == cycle[1] || adj_list[cycle[end]][2] == cycle[1]
        return cycle
    else
        return nothing
    end
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
    visited = falses(length(connectivity))
    parents = zeros(Int, length(connectivity))
    basic_loops = []
    while !all(visited)
        vertex = findfirst(!, visited)
        find_loops!(connectivity, vertex, parents, visited, basic_loops)
    end

    println("basic loops:", length(basic_loops))

    n = length(basic_loops)
    all_combinations = [[x...] for k in 2:n for x in combinations(1:n, k)]

    println("total combinations: ", length(all_combinations))

    all_loops = basic_loops
    for combo in all_combinations
        new_loop = add_loops(basic_loops[combo])
        if !isnothing(new_loop)
            push!(all_loops, new_loop)
        end
    end

    println("total loops:", length(all_loops))

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

    println("no chord loops:", length(no_chord_loops))

    facial_loops = []
    for loop in no_chord_loops
        if is_loop_facial(loop, connectivity)
            push!(facial_loops, loop)
        end
    end

    return facial_loops
end

app_cnt = 18
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_geometry/app_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
abstract_edges, connectivity = conectivity_analysis(vertices, edges)
@time abstract_faces = find_faces(connectivity)
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
