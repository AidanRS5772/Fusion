using JSON3: Error
using Core: OptimizedGenerics
using JSON3
using Gmsh
using LinearAlgebra
using Combinatorics

function circular_sort(axis, v0, vecs)
    b1 = cross(axis, [0, 0, 1])
    if norm(b1) < 1e-6
        b1 = cross(axis, [1, 0, 0])
    end
    normalize!(b1)
    b2 = cross(axis, b1)

    projs = []
    for v in vecs
        p = (v .- v0) .- dot(v .- v0, axis) .* axis
        push!(projs, (dot(p, b1), dot(p, b2)))
    end

    angles = [atan(x, y) for (x, y) in projs]
    return sortperm(angles)
end

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
        perm = circular_sort(normalize(vertices[i]), vertices[i], [vertices[a] for a in adj])

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
    geo_ellipse::Int
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

function make_intersection_ellipses(model, mesh_size, vertices, geo_vertices, adj_list, radius)
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

            geo_ell = model.geo.addEllipseArc(geo_M, geo_v0, geo_M, geo_m)
            push!(ellipses, Ellipse(v0, geo_v0, M, geo_M, m, geo_m, geo_ell))
        end
        push!(all_ellipses, ellipses)
    end
    return all_ellipses
end

function make_ellipse_from_points(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2)
    t1 = dot(p1 .- vertex, axis)
    t2 = dot(p2 .- vertex, axis)
    tm = (t1 + t2) / 2
    c = vertex .+ tm .* axis
    n = normalize(cross(p1 .- c, p2 .- c))
    m = normalize(cross(n, axis)) .* radius
    M = normalize(cross(m, n)) .* (radius / dot(n, axis))

    mid = Nothing
    k = -tm / dot(M, axis)
    if abs(k) < 1
        mid = c .+ M .* k .+ m .* sqrt(1 - k^2)
        if dot(mid .- c, p1 .- c) < 0 && dot(mid .- c, p2 .- c) < 0
            mid .-= 2 .* m .* sqrt(1 - k^2)
        end
    else
        mid = normalize(normalize(p1 .- c) .+ normalize(p2 .- c)) .* radius .+ c
    end

    geo_c = model.geo.addPoint(c[1], c[2], c[3], mesh_size)
    geo_mid = model.geo.addPoint(mid[1], mid[2], mid[3], mesh_size)
    Mp = c .+ normalize(M) .* radius
    geo_M = model.geo.addPoint(Mp[1], Mp[2], Mp[3], mesh_size)

    ell1 = model.geo.addEllipseArc(geo_p1, geo_c, geo_M, geo_mid)
    ell2 = model.geo.addEllipseArc(geo_mid, geo_c, geo_M, geo_p2)

    return ell1, ell2, geo_mid, mid
end

function make_line_or_ellipse(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2, err=0.001)
    if norm(cross(normalize(p1 .- p2), axis)) < err
        l = normalize(p1 .- p2)
        p = (p1 .+ p2) ./ 2
        mid = p .+ (dot(vertex .- p, axis) / dot(l, axis)) .* l
        geo_mid = model.geo.addPoint(mid[1], mid[2], mid[3], mesh_size)
        line1 = model.geo.addLine(geo_p1, geo_mid)
        line2 = model.geo.addLine(geo_mid, geo_p2)
        return line1, line2, geo_mid, mid
    else
        return make_ellipse_from_points(model, mesh_size, vertex, axis, radius, p1, p2, geo_p1, geo_p2)
    end
end

function make_edge_wires(model, mesh_size, vertices, geo_vertices, edges, adj_list, inter_ells, radius)
    surfaces = []
    outer_curves = [[] for _ in 1:length(vertices)]
    inner_curves = [[] for _ in 1:length(vertices)]
    for (e1, e2) in edges
        edge_surfaces = []
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
        geo_end_cir1 = model.geo.addCircleArc(geo_minor1_p1, geo_vertex1, geo_minor1_p2)
        push!(outer_curves[e1], (geo_end_cir1, geo_minor1_p1, geo_minor1_p2))

        vertex2 = vertices[e2]
        geo_vertex2 = geo_vertices[e2]
        minor2_p1 = inter_ells[e2][adj_idx2].minor .+ vertex2
        geo_minor2_p1 = inter_ells[e2][adj_idx2].geo_minor
        minor2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].minor .+ vertex2
        geo_minor2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].geo_minor
        geo_end_cir2 = model.geo.addCircleArc(geo_minor2_p1, geo_vertex2, geo_minor2_p2)
        push!(outer_curves[e2], (geo_end_cir2, geo_minor2_p1, geo_minor2_p2))

        #make intersection ellipse arcs
        major1_p1 = inter_ells[e1][adj_idx1].major .+ vertex1
        geo_major1_p1 = inter_ells[e1][adj_idx1].geo_major
        geo_inter1_ell1 = inter_ells[e1][adj_idx1].geo_ellipse

        major1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].major .+ vertex1
        geo_major1_p2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].geo_major
        geo_inter1_ell2 = inter_ells[e1][mod1(adj_idx1 - 1, n1)].geo_ellipse

        major2_p1 = inter_ells[e2][adj_idx2].major .+ vertex2
        geo_major2_p1 = inter_ells[e2][adj_idx2].geo_major
        geo_inter2_ell1 = inter_ells[e2][adj_idx2].geo_ellipse

        major2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].major .+ vertex2
        geo_major2_p2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].geo_major
        geo_inter2_ell2 = inter_ells[e2][mod1(adj_idx2 - 1, n2)].geo_ellipse

        axis = normalize(vertices[e1] .- vertices[e2])
        mid_vertex = (vertices[e1] .+ vertices[e2]) ./ 2
        geo_mid_vertex = model.geo.addPoint(mid_vertex[1], mid_vertex[2], mid_vertex[3], mesh_size)

        #make interior cylinder ellipses
        geo_cyl1_ell1, geo_cyl1_ell2, geo_cyl1_mid, cyl1_mid = make_ellipse_from_points(model, mesh_size, vertex1, axis, radius, major1_p1, major1_p2, geo_major1_p1, geo_major1_p2)
        push!(inner_curves[e1], (geo_cyl1_ell1, geo_cyl1_ell2, cyl1_mid, geo_cyl1_mid))
        geo_cyl2_ell1, geo_cyl2_ell2, geo_cyl2_mid, cyl2_mid = make_ellipse_from_points(model, mesh_size, vertex2, axis, radius, major2_p1, major2_p2, geo_major2_p1, geo_major2_p2)
        push!(inner_curves[e2], (geo_cyl2_ell1, geo_cyl2_ell2, cyl2_mid, geo_cyl2_mid))

        #make cross ellipse connection
        geo_con1_l1, geo_con1_l2, geo_con1_mid, con1_mid = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, minor1_p2, minor2_p1, geo_minor1_p2, geo_minor2_p1)
        geo_con2_l1, geo_con2_l2, geo_con2_mid, con2_mid = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, minor1_p1, minor2_p2, geo_minor1_p1, geo_minor2_p2)
        geo_con3_l1, geo_con3_l2, geo_con3_mid, con3_mid = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, major1_p2, major2_p1, geo_major1_p2, geo_major2_p1)
        geo_con4_l1, geo_con4_l2, geo_con4_mid, con4_mid = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, major1_p1, major2_p2, geo_major1_p1, geo_major2_p2)
        geo_con5_l1, geo_con5_l2, geo_con5_mid, con5_mid = make_line_or_ellipse(model, mesh_size, mid_vertex, axis, radius, cyl1_mid, cyl2_mid, geo_cyl1_mid, geo_cyl2_mid)

        #make middle circle arcs
        geo_mid_cir1 = model.geo.addCircleArc(geo_con1_mid, geo_mid_vertex, geo_con2_mid)
        geo_mid_cir2 = model.geo.addCircleArc(geo_con5_mid, geo_mid_vertex, geo_con3_mid)
        geo_mid_cir3 = model.geo.addCircleArc(geo_con5_mid, geo_mid_vertex, geo_con4_mid)
        geo_mid_cir4 = model.geo.addCircleArc(geo_con2_mid, geo_mid_vertex, geo_con4_mid)
        geo_mid_cir5 = model.geo.addCircleArc(geo_con1_mid, geo_mid_vertex, geo_con3_mid)

        #make top pannel
        top_curve1 = model.geo.addCurveLoop([geo_end_cir1, geo_con1_l1, geo_mid_cir1, geo_con2_l1], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([top_curve1]))
        top_curve2 = model.geo.addCurveLoop([geo_end_cir2, geo_con1_l2, geo_mid_cir1, geo_con2_l2], -1, true)
        push!(surfaces, model.geo.addSurfaceFilling([top_curve2]))

        #make bottom pannel
        bottom1_curve1 = model.geo.addCurveLoop([geo_cyl1_ell1, geo_con5_l1, geo_mid_cir3, geo_con4_l1], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([bottom1_curve1]))
        bottom1_curve2 = model.geo.addCurveLoop([geo_cyl1_ell2, geo_con3_l1, geo_mid_cir2, geo_con5_l1], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([bottom1_curve2]))
        bottom2_curve1 = model.geo.addCurveLoop([geo_cyl2_ell1, geo_con5_l2, geo_mid_cir2, geo_con3_l2], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([bottom2_curve1]))
        bottom2_curve2 = model.geo.addCurveLoop([geo_cyl2_ell2, geo_con4_l2, geo_mid_cir3, geo_con5_l2], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([bottom2_curve2]))

        #make side pannels
        side1_curve1 = model.geo.addCurveLoop([geo_inter1_ell1, geo_con2_l1, geo_mid_cir4, geo_con4_l1], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([side1_curve1]))
        side1_curve2 = model.geo.addCurveLoop([geo_inter2_ell2, geo_con4_l2, geo_mid_cir4, geo_con2_l2], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([side1_curve2]))
        side2_curve1 = model.geo.addCurveLoop([geo_inter1_ell2, geo_con1_l1, geo_mid_cir5, geo_con3_l1], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([side2_curve1]))
        side2_curve2 = model.geo.addCurveLoop([geo_inter2_ell1, geo_con3_l2, geo_mid_cir5, geo_con1_l2], -1, true)
        push!(edge_surfaces, model.geo.addSurfaceFilling([side2_curve2]))

        push!(surfaces, edge_surfaces)
    end

    return surfaces, outer_curves, inner_curves
end

function make_outer_surfaces(model, mesh_size, verticies, geo_verticies, all_outer_curves, radius)
    outer_surfaces = []
    for (i, curve_info) in enumerate(all_outer_curves)
        if length(curve_info) == 3
            loop = model.geo.addCurveLoop([curve for (curve, _, _) in curve_info], -1, true)
            push!(outer_surfaces, Tuple(model.geo.addSurfaceFilling([loop])))
        else
            geo_vertex = geo_verticies[i]
            common = verticies[i] .* (1 + radius / norm(verticies[i]))
            geo_common = model.geo.addPoint(common[1], common[2], common[3], mesh_size)
            surfaces = []
            for (curve, geo_p1, geo_p2) in curve_info
                c1 = model.geo.addCircleArc(geo_p1, geo_vertex, geo_common)
                c2 = model.geo.addCircleArc(geo_p2, geo_vertex, geo_common)
                loop = model.geo.addCurveLoop([curve, c1, c2], -1, true)
                push!(surfaces, model.geo.addSurfaceFilling([loop]))
            end
            push!(outer_surfaces, surfaces)
        end
    end

    return outer_surfaces
end

function find_loop(model, arcs1, arcs2, arc3)
    for arc1 in arcs1
        b1 = [tag for (_, tag) in model.getBoundary([(1, arc1)])]
        for arc2 in arcs2
            b2 = [tag for (_, tag) in model.getBoundary([(1, arc2)])]
            if !isempty(intersect(b1, b2))
                return model.geo.addCurveLoop([arc1, arc2, arc3], -1, true)
            end
        end
    end
end

function make_inner_surfaces(model, mesh_size, verticies, all_inner_curves)
    inner_surfaces = []
    for (i, curve_info) in enumerate(all_inner_curves)
        surfaces = []
        if length(curve_info) == 3
            curve_info1, curve_info2, curve_info3 = curve_info
            curve1_ell1, curve1_ell2, _, geo_mid1, = curve_info1
            curve2_ell1, curve2_ell2, _, geo_mid2 = curve_info2
            curve3_ell1, curve3_ell2, _, geo_mid3 = curve_info3

            line1 = model.geo.addLine(geo_mid1, geo_mid2)
            line2 = model.geo.addLine(geo_mid2, geo_mid3)
            line3 = model.geo.addLine(geo_mid3, geo_mid1)

            line_loop = model.geo.addCurveLoop([line1, line2, line3], -1, true)
            push!(surfaces, model.geo.addPlaneSurface([line_loop]))

            loop1 = find_loop(model, [curve1_ell1, curve1_ell2], [curve2_ell1, curve2_ell2], line1)
            push!(surfaces, model.geo.addSurfaceFilling([loop1]))

            loop2 = find_loop(model, [curve2_ell1, curve2_ell2], [curve3_ell1, curve3_ell2], line2)
            push!(surfaces, model.geo.addSurfaceFilling([loop2]))

            loop3 = find_loop(model, [curve3_ell1, curve3_ell2], [curve1_ell1, curve1_ell2], line3)
            push!(surfaces, model.geo.addSurfaceFilling([loop3]))
        else
            mids = [mid for (_, _, mid, _) in curve_info]
            perm = circular_sort(normalize(vertices[i]), vertices[i], mids)
            sorted_curve_info = curve_info[perm]
            n = length(curve_info)
            lines = [model.geo.addLine(sorted_curve_info[j][4], sorted_curve_info[mod1(j + 1, n)][4]) for j in 1:n]
            loops = [find_loop(model, sorted_curve_info[j][1:2], sorted_curve_info[mod1(j + 1, n)][1:2], lines[j]) for j in 1:n]
            append!(surfaces, [model.geo.addSurfaceFilling([loop]) for loop in loops])
            common = sum(mids) / length(mids)
            geo_common = model.geo.addPoint(common[1], common[2], common[3], mesh_size)
            for j in 1:n
                l1 = model.geo.addLine(sorted_curve_info[j][4], geo_common)
                l2 = model.geo.addLine(sorted_curve_info[mod1(j + 1, n)][4], geo_common)
                loop = model.geo.addCurveLoop([l1, l2, lines[j]], -1, true)
                push!(surfaces, model.geo.addPlaneSurface([loop]))
            end
        end
        push!(inner_surfaces, surfaces)
    end

    return inner_surfaces
end

function flatten_tags(item)
    result = Int32[]

    if item isa Tuple
        push!(result, Int32(item[1]))
    elseif item isa Number
        push!(result, Int32(item))
    elseif item isa Vector || item isa Tuple
        for subitem in item
            append!(result, flatten_tags(subitem))
        end
    end

    return result
end

app_cnt = 42
cathode_radius = 0.05
anode_radius = 0.25
wire_radius = 0.0025

json_data = JSON3.read("cathode_data/appratures_$(app_cnt).json")
edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
vertices = Vector{Vector{Float64}}(json_data["vertices"])
face_centers = Vector{Vector{Float64}}(json_data["points"])
abstract_edges, adj_list = conectivity_analysis(vertices, edges)

scaled_anode_radius = anode_radius / cathode_radius
scaled_wire_radius = wire_radius / cathode_radius

try
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    #gmsh.option.setNumber("Geometry.Points", 0)
    model = gmsh.model
    model.add("Fusion")
    mesh_size = scaled_wire_radius / 4

    geo_vertices = [model.geo.addPoint(x, y, z, mesh_size) for (x, y, z) in vertices]
    #geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]

    inter_ells = make_intersection_ellipses(model, mesh_size, vertices, geo_vertices, adj_list, scaled_wire_radius)
    surfaces, outer_curves, inner_curves = make_edge_wires(model, mesh_size, vertices, geo_vertices, abstract_edges, adj_list, inter_ells, scaled_wire_radius)
    model.geo.synchronize()

    outer_surfaces = make_outer_surfaces(model, mesh_size, vertices, geo_vertices, outer_curves, scaled_wire_radius)
    inner_surfaces = make_inner_surfaces(model, mesh_size, vertices, inner_curves)
    model.geo.synchronize()

    all_surface_tags = Int32[]
    append!(all_surface_tags, flatten_tags(surfaces))
    append!(all_surface_tags, flatten_tags(outer_surfaces))
    append!(all_surface_tags, flatten_tags(inner_surfaces))
    cathode = model.geo.addSurfaceLoop(all_surface_tags)
    model.geo.synchronize()

    model.mesh.generate(2)
    gmsh.fltk.run()
finally
    gmsh.finalize()
end
