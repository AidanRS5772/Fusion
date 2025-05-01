using JSON3: Error
using Core: OptimizedGenerics
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

        sym_axsis = vertices[i]
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

struct Ellipse
    center::Vector{Float64}
    geo_center::Int
    major::Vector{Float64}
    geo_major::Int
    minor::Vector{Float64}
    geo_minor::Int
end

function find_intersection_ellipse(v0, e1, e2, radius)
    m = cross(e1, e2)
    if dot(m, v0) < 0
        m .*= -1
    end
    M = e1 .+ e2
    M .*= radius / norm(m)
    normalize!(m)
    m .*= radius

    return M, m
end

function intersection_ellipses(model, mesh_size, vertices, geo_vertices, adj_list, radius)
    all_ellipses = []
    for (i, adjs) in enumerate(adj_list)
        ellipses = []
        n = length(adjs)
        v0 = vertices[i]
        geo_v0 = geo_vertices[i]
        for i in 1:n
            v1 = vertices[adjs[i]]
            v2 = vertices[adjs[mod1(i + 1, n)]]

            e1 = normalize(v1 - v0)
            e2 = normalize(v2 - v0)

            M, m = find_intersection_ellipse(v0, e1, e2, radius)

            geo_M = model.geo.addPoint(M[1] + v0[1], M[2] + v0[2], M[3] + v0[3], mesh_size)
            geo_m = model.geo.addPoint(m[1] + v0[1], m[2] + v0[2], m[3] + v0[3], mesh_size)
            push!(ellipses, Ellipse(v0, geo_v0, M, geo_M, m, geo_m))
        end
        push!(all_ellipses, ellipses)
    end
    return all_ellipses
end

function make_ellipse_from_points(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2)
    t1 = dot(p1 .- vertex, axis)
    t2 = dot(p2 .- vertex, axis)
    tm = ((t1 + t2) / 2)
    c = vertex .+ tm .* axis
    mid = normalize(normalize(p1 .- c) .+ normalize(p2 .- c)) .* radius .+ c
    n = normalize(cross(p1 .- c, p2 .- c))
    m = normalize(cross(n, axis))
    M = normalize(cross(m, axis)) .+ c

    geo_c = model.geo.addPoint(c[1], c[2], c[3], mesh_size)
    geo_mid = model.geo.addPoint(mid[1], mid[2], mid[3], mesh_size)
    geo_M = model.geo.addPoint(M[1], M[2], M[3], mesh_size)

    ell1 = model.geo.addEllipseArc(geo_p1, geo_c, geo_M, geo_mid)
    ell2 = model.geo.addEllipseArc(geo_mid, geo_c, geo_M, geo_p2)

    return ell1, ell2, geo_mid
end

function make_line_or_ellipse(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2, err=0.001)
    if norm(cross(normalize(p1 .- p2), axis)) < err
        l = normalize(p1 .- p2)
        p = (p1 .+ p2) ./ 2
        mid = p .+ dot((vertex .- p), axis) .* l
        geo_mid = model.geo.addPoint(mid[1], mid[2], mid[3], mesh_size)
        line1 = model.geo.addLine(geo_p1, geo_mid)
        line2 = model.geo.addLine(geo_mid, geo_p2)
        return line2, line1, geo_mid
    else
        return make_ellipse_from_points(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2)
    end
end

function reverse_orientation(tags)
    return Tuple(collect(tags) .* -1)
end

function edge_wires(model, mesh_size, vertices, geo_vertices, edges, adj_list, inter_ells, radius)
    for (e1, e2) in edges
        n1 = length(adj_list[e1])
        n2 = length(adj_list[e2])
        adj_idx1 = findfirst(x -> x == e2, adj_list[e1])
        adj_idx2 = findfirst(x -> x == e1, adj_list[e2])

        #make circle arcs on the exterior of intersections
        vertex1 = vertices[e1]
        geo_vertex1 = geo_vertices[e1]
        minor1_p1 = inter_ells[e1][adj_idx1].minor .+ vertex1
        geo_minor1_p1 = inter_ells[e1][adj_idx1].geo_minor
        minor1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].minor .+ vertex1
        geo_minor1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].geo_minor
        geo_cir1 = model.geo.addCircleArc(geo_minor1_p1, geo_vertex1, geo_minor1_p2)

        vertex2 = vertices[e2]
        geo_vertex2 = geo_vertices[e2]
        minor2_p1 = inter_ells[e2][adj_idx2].minor .+ vertex2
        geo_minor2_p1 = inter_ells[e2][adj_idx2].geo_minor
        minor2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].minor .+ vertex2
        geo_minor2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].geo_minor
        geo_cir2 = model.geo.addCircleArc(geo_minor2_p1, geo_vertex2, geo_minor2_p2)

        #make intersection ellipse arcs
        major1_p1 = inter_ells[e1][adj_idx1].major .+ vertex1
        geo_major1_p1 = inter_ells[e1][adj_idx1].geo_major
        geo_inter1_ell1 = model.geo.addEllipseArc(geo_major1_p1, geo_vertex1, geo_major1_p1, geo_minor1_p1)

        major1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].major .+ vertex1
        geo_major1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].geo_major
        geo_inter1_ell2 = model.geo.addEllipseArc(geo_major1_p2, geo_vertex1, geo_major1_p2, geo_minor1_p2)

        major2_p1 = inter_ells[e2][adj_idx2].major .+ vertex2
        geo_major2_p1 = inter_ells[e2][adj_idx2].geo_major
        geo_inter2_ell1 = model.geo.addEllipseArc(geo_major2_p1, geo_vertex2, geo_major2_p1, geo_minor2_p1)

        major2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].major .+ vertex2
        geo_major2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].geo_major
        geo_inter2_ell2 = model.geo.addEllipseArc(geo_major2_p2, geo_vertex2, geo_major2_p2, geo_minor2_p2)

        #make cross ellipse connections
        axis = normalize(vertices[e1] .- vertices[e2])
        mid_vertex = (vertices[e1] .+ vertices[e2]) ./ 2

        #need to make intermediat connections between mid points
        geo_con1 = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, minor1_p2, minor2_p1, geo_minor1_p2, geo_minor2_p1)
        geo_con2 = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, minor1_p1, minor2_p2, geo_minor1_p1, geo_minor2_p2)
        geo_con3 = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, major1_p2, major2_p1, geo_major1_p2, geo_major2_p1)
        geo_con4 = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, major1_p1, major2_p2, geo_major1_p1, geo_major2_p2)

        #make interior cylinder ellipses
        geo_cyl1 = make_ellipse_from_points(model, mesh_size, vertex1, axis, radius, major1_p1, major1_p2, geo_major1_p1, geo_major1_p2)
        geo_cyl2 = make_ellipse_from_points(model, mesh_size, vertex2, axis, radius, major2_p1, major2_p2, geo_major2_p1, geo_major2_p2)

        # #make top pannel
        # top_curve = model.geo.addCurveLoop([geo_cir1, geo_con1..., geo_cir2, reverse_orientation(geo_con2)...])
        # top_panel = model.geo.addSurfaceFilling([top_curve])

        # #make bottom pannel
        # bottom_curve = model.geo.addCurveLoop([geo_cyl1..., geo_con3..., geo_cyl2..., reverse_orientation(geo_con4)...])
        # bottom_panel = model.geo.addSurfaceFilling([bottom_curve])
    end
end

app_cnt = 10
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.005

json_data = JSON3.read("cathode_data/appratures_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
face_centers = Vector{Vector{Float64}}(json_data["points"])
abstract_edges, adj_list = conectivity_analysis(vertices, edges)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

try
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    #gmsh.option.setNumber("Geometry.Points", 0)
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 8

    geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    #geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]

    inter_ells = intersection_ellipses(model, mesh_size, vertices, geo_vertices, adj_list, scaled_wire_radius)
    edge_wires(model, mesh_size, vertices, geo_vertices, abstract_edges, adj_list, inter_ells, scaled_wire_radius)

    model.geo.synchronize()
    model.mesh.generate(2)
    gmsh.fltk.run()
finally
    gmsh.finalize()
end
