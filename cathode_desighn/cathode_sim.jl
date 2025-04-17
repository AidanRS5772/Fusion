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

function find_basis_loops!(basis_loops, visited_vertices, visited_edges, parents, v, adj_list)
    visited_vertices[v] = true
    for adj in adj_list[v]
        edge = (min(v, adj), max(v, adj))
        if (adj != parents[v]) && !(edge in visited_edges)
            push!(visited_edges, edge)
            if visited_vertices[adj]
                v_path = []
                cur = v
                while cur != 0
                    push!(v_path, cur)
                    cur = parents[cur]
                end

                adj_path = []
                cur = adj
                while !(cur in v_path)
                    pushfirst!(adj_path, cur)
                    cur = parents[cur]
                end

                FCA_idx = findfirst(x -> x == cur, v_path)
                loop = [v_path[1:FCA_idx]; adj_path]
                push!(basis_loops, Tuple(loop))
            else
                parents[adj] = v
                find_basis_loops!(basis_loops, visited_vertices, visited_edges, parents, adj, adj_list)
            end
        end
    end
end

function add_loops(loop_edge_sets)
    new_loop_edge_set = symdiff(loop_edge_sets...)

    visited_edges = falses(length(new_loop_edge_set))
    visited_edges[1] = true
    start = new_loop_edge_set[1][1]
    path = [start]
    prev = start
    cur = new_loop_edge_set[1][2]

    while cur != start
        push!(path, cur)
        edge_idxs = findall(x -> (cur == x[1] || cur == x[2]) && prev != x[1] && prev != x[2], new_loop_edge_set)
        if length(edge_idxs) > 1 || isnothing(edge_idxs)
            return nothing
        end
        edge_idx = edge_idxs[1]
        visited_edges[edge_idx] = true
        prev = cur
        cur = new_loop_edge_set[edge_idx][1]
        if cur == prev
            cur = new_loop_edge_set[edge_idx][2]
        end
    end

    if all(visited_edges)
        return Tuple(path)
    else
        return nothing
    end
end

function find_proper_loops(adj_list)
    basis_loops = []
    visited_vertices = falses(length(adj_list))
    visited_edges = Set()
    parents = zeros(Int, length(adj_list))
    for v in eachindex(adj_list)
        if !visited_vertices[v]
            find_basis_loops!(basis_loops, visited_vertices, visited_edges, parents, v, adj_list)
        end
    end

    if isempty(basis_loops)
        return []
    end

    all_loops = basis_loops
    if length(basis_loops) != 1
        basis_loop_edge_lists = []
        for loop in basis_loops
            edges = []
            n = length(loop)
            for i in 0:n-1
                v1 = loop[i+1]
                v2 = loop[mod(i + 1, n)+1]
                push!(edges, (min(v1, v2), max(v1, v2)))
            end
            push!(basis_loop_edge_lists, edges)
        end

        for combo in combinations(1:length(basis_loops))
            if length(combo) != 1
                new_loop = add_loops(basis_loop_edge_lists[combo])
                if !isnothing(new_loop)
                    push!(all_loops, new_loop)
                end
            end
        end
    end

    no_chord_loops = []
    for loop in all_loops
        n = length(loop)
        chord = false
        for i in 0:n-1
            v = loop[i+1]
            next = loop[mod(i + 1, n)+1]
            prev = loop[mod(i - 1, n)+1]
            for adj in adj_list[v]
                if adj != next && adj != prev
                    if adj in loop
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

    return no_chord_loops
end

function is_facial(loop, adj_list)
    n = length(adj_list)
    visited = falses(n)
    visited[loop] .= true
    start = findfirst(!, visited)
    queue = [start]
    while !isempty(queue)
        cur = popfirst!(queue)
        visited[cur] = true
        if all(visited)
            return true
        end
        for adj in adj_list[cur]
            if !(adj in loop) && !(visited[adj])
                push!(queue, adj)
            end
        end
    end

    return false
end

function form_sub_graph(adj_list, sub_vertices)
    new_adj_list = []
    for v in sub_vertices
        adj = []
        for u in adj_list[v]
            sub_u = findfirst(x -> u == x, sub_vertices)
            if !isnothing(sub_u)
                push!(adj, sub_u)
            end
        end
        push!(new_adj_list, adj)
    end

    return new_adj_list
end

function find_faces(vertices, face_centers, adj_list)
    sphere_dist(v, fc) = acos(dot(normalize(v), normalize(fc)))
    sorted_vertex_lists = []
    base_vertices = 1:length(vertices)
    for fc in face_centers
        sphere_dist_list = [acos(dot(normalize(v), normalize(fc))) for v in vertices]
        perm = sortperm(sphere_dist_list)
        push!(sorted_vertex_lists, base_vertices[perm])
    end

    faces = []
    for vertex_list in sorted_vertex_lists
        found_face = false
        for i in 3:length(vertex_list)
            sub_adj_list = form_sub_graph(adj_list, vertex_list[1:i])
            sub_loops = find_proper_loops(sub_adj_list)
            for sub_loop in sub_loops
                loop = vertex_list[[sub_loop...]]
                if is_facial(loop, adj_list)
                    min_idx = argmin(loop)
                    if min_idx != 1
                        loop = [loop[min_idx:end]; loop[1:min_idx-1]]
                    end
                    loop = Tuple(loop)
                    if !(loop in faces)
                        found_face = true
                        push!(faces, Tuple(loop))
                        break
                    end
                end
            end
            if found_face
                break
            end
        end
    end

    return faces
end

function add_bisections(model, mesh_size, vertices, faces, radius)
    all_bisects = []
    all_geo_bisects = []
    for face in faces
        bisects = []
        geo_bisects = []
        n = length(face)
        for i in 1:n
            v0 = vertices[face[mod1(i - 1, n)]]
            v1 = vertices[face[i]]
            v2 = vertices[face[mod1(i + 1, n)]]

            e1 = normalize(v0 .- v1)
            e2 = normalize(v2 .- v1)

            b = e1 .+ e2
            if dot(b, e1) < 0 || dot(b, e2) < 0
                b .*= -1
            end
            b .*= radius / norm(cross(e1, e2))
            b .+= v1

            push!(bisects, b)
            push!(geo_bisects, model.geo.addPoint(b[1], b[2], b[3], mesh_size))
        end
        for i in 1:n
            model.geo.addLine(geo_bisects[i], geo_bisects[mod1(i + 1, n)])
        end
        push!(all_bisects, bisects)
        push!(all_geo_bisects, geo_bisects)
    end

    return all_geo_bisects, all_bisects
end

struct Ellipse
    center::Vector{Float64}
    major::Vector{Float64}
    minor::Vector{Float64}
end

function make_ellipse_bisects(vertex, axis, radius, b1, b2)
    cyl1 = normalize(cross(axis, vertex))
    cyl2 = normalize(cross(axis, cyl1))

    t1 = dot(b1, axis)
    t2 = dot(b2, axis)
    θ1 = atan(dot(b1, cyl1), dot(b1, cyl2))
    θ2 = atan(dot(b2, cyl1), dot(b2, cyl2))

    b0 = vertex .+ axis .* (t1 + t2) ./ 2 .+ radius .* (cyl1 .* cos((θ1 + θ2) / 2) .+ cyl2 .* sin((θ1 + θ2) / 2))
    normal = normalize(cross(b1 .- b0, b2 .- b0))
    if dot(normal, axis) < 0
        normal .*= -1
    end

    center = (dot(b0 .- vertex, normal) / dot(axis, normal)) .* axis .+ vertex
    minor = normalize(cross(normal, axis)) .* radius
    major = normalize(cross(minor, normal)) ./ dot(normal, axis)
    return Ellipse(center, major, minor)
end

function make_full_geo_ellipse(model, mesh_size, ellipse)
    geo_center = model.geo.addPoint(ellipse.center[1], ellipse.center[2], ellipse.center[3], mesh_size)

    Mp1 = ellipse.center .+ ellipse.major
    geo_Mp1 = model.geo.addPoint(Mp1[1], Mp1[2], Mp1[3], mesh_size)
    Mp2 = ellipse.center .- ellipse.major
    geo_Mp2 = model.geo.addPoint(Mp2[1], Mp2[2], Mp2[3], mesh_size)
    mp1 = ellipse.center .+ ellipse.minor
    geo_mp1 = model.geo.addPoint(mp1[1], mp1[2], mp1[3], mesh_size)
    mp2 = ellipse.center .- ellipse.minor
    geo_mp2 = model.geo.addPoint(mp2[1], mp2[2], mp2[3], mesh_size)

    e1 = model.geo.addEllipseArc(geo_Mp1, geo_center, geo_Mp1, geo_mp1)
    e2 = model.geo.addEllipseArc(geo_mp1, geo_center, geo_Mp1, geo_Mp2)
    e3 = model.geo.addEllipseArc(geo_Mp2, geo_center, geo_Mp1, geo_mp2)
    e4 = model.geo.addEllipseArc(geo_mp2, geo_center, geo_Mp1, geo_Mp1)

    curve_loop = model.geo.addCurveLoop([e1, e2, e3, e4])
    return curve_loop
end

function make_edge_face_list(edges, faces)
    edge_face_list = [Vector{Int}() for _ in 1:length(edges)]
    for (face_idx, face) in enumerate(faces)
        n = length(face)
        for i in 1:n
            v1 = face[i]
            v2 = face[mod1(i + 1, n)]
            edge = Tuple([min(v1, v2), max(v1, v2)])
            edge_idx = findfirst(x -> x == edge, edges)
            push!(edge_face_list[edge_idx], face_idx)
        end
    end

    return edge_face_list
end

function add_bisect_ellipses(model, mesh_size, vertices, radius, bisects, adj_list, edges, faces, edge_face_list)
    all_bisect_ellipses = []
    all_geo_bisect_ellipses = []
    for (i, adjs) in enumerate(adj_list)
        bisect_ellipses = []
        geo_bisect_ellipses = []
        vertex = vertices[i]
        for adj in adjs
            edge = Tuple([min(i, adj), max(i, adj)])
            edge_idx = findfirst(x -> x == edge, edges)
            if isnothing(edge_idx)
                throw(ErrorException("Edge not found in edges of $(edge)"))
            end
            faces_idx1, faces_idx2 = edge_face_list[edge_idx]
            face_idx1 = findfirst(x -> x == i, faces[faces_idx1])
            face_idx2 = findfirst(x -> x == i, faces[faces_idx2])

            b1 = bisects[faces_idx1][face_idx1]
            b2 = bisects[faces_idx2][face_idx2]
            axis = normalize(vertices[adj] .- vertex)

            ellipse = make_ellipse_bisects(vertex, axis, radius, b1, b2)
            geo_ellipse = make_full_geo_ellipse(model, mesh_size, ellipse)

            push!(bisect_ellipses, ellipse)
            push!(geo_bisect_ellipses, geo_ellipse)
        end
        push!(all_bisect_ellipses, bisect_ellipses)
        push!(all_geo_bisect_ellipses, geo_bisect_ellipses)
    end

    return all_geo_bisect_ellipses, all_bisect_ellipses
end

app_cnt = 6
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_data/appratures_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
face_centers = Vector{Vector{Float64}}(json_data["points"])
abstract_edges, adj_list = conectivity_analysis(vertices, edges)
abstract_faces = find_faces(vertices, face_centers, adj_list)
edge_face_list = make_edge_face_list(abstract_edges, abstract_faces)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

display(vertices)
display(abstract_edges)
display(adj_list)
display(abstract_faces)
display(edge_face_list)

try
    gmsh.initialize()
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 8

    geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]
    geo_bisects, bisects = add_bisections(model, mesh_size, vertices, abstract_faces, scaled_wire_radius)

    model.geo.synchronize()
    model.mesh.generate(1)
    gmsh.fltk.run()
catch e
    println("Error: ", e)
    stacktrace()
finally
    gmsh.finalize()
end
