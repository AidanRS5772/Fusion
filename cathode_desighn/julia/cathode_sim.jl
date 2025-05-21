using LinearAlgebra
using JSON3
using Gmsh
using PlotlyJS
using Gridap
using GridapGmsh
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

function find_loop!(adj_list, loop, visited, v)
    visited[v] = true
    push!(loop, v)
    for adj in adj_list[v]
        if visited[adj]
            if loop[1] == adj && loop[end-1] != adj
                return true
            end
        else
            if find_loop!(adj_list, loop, visited, adj)
                return true
            end
        end
    end
    pop!(loop)
    return false
end

function find_abstract_faces(vertices, face_centers, adj_list)
    faces = []
    for face_center in face_centers
        angles = [acos(dot(normalize(face_center), normalize(vertex))) for vertex in vertices]
        perm = sortperm(angles)
        for i in 3:length(perm)
            new_adj_list = []
            for adjs in adj_list[perm[1:i]]
                new_adjs = []
                for adj in adjs
                    idx = findfirst(isequal(adj), perm[1:i])
                    if !isnothing(idx)
                        push!(new_adjs, idx)
                    end
                end
                push!(new_adj_list, new_adjs)
            end

            loop = []
            visited = falses(i)
            found = find_loop!(new_adj_list, loop, visited, 1)
            if found
                push!(faces, perm[loop])
                break
            end
        end
    end

    return faces
end

function find_triangulations!(found_loops, loop)
    triangulations = []
    if !(loop in found_loops)
        n = length(loop)
        if n == 3
            return [loop]
        end

        for i in 1:(n-2)
            for j in (i+2):(n+i-2-max(i - 2, 0))
                left = loop[i:j]
                right = [loop[j:end]..., loop[1:i]...]
                left_triangles = find_triangulations!(found_loops, left)
                right_triangles = find_triangulations!(found_loops, right)
                push!(found_loops, left, right)
                append!(triangulations, left_triangles)
                append!(triangulations, right_triangles)
            end
        end
    end

    return triangulations
end

function find_inner_bound(vertices, faces, radius)
    all_triangles = []
    for face in faces
        found_loops = Set()
        triangles = find_triangulations!(found_loops, face)
        append!(all_triangles, triangles)
    end

    min_norms = []
    for tri in all_triangles
        a = vertices[tri[1]]
        b = vertices[tri[2]]
        c = vertices[tri[3]]

        e1 = a - c
        e2 = b - c
        e3 = a - b
        E = [e1 e2]
        t = -(E' * E) \ (E' * c)
        if (t[1] < 0 || t[2] < 0 || (t[1] + t[2]) > 1)
            norms = []

            t1 = -dot(e1, c) / dot(e1, e1)
            if (0 < t1 < 1)
                push!(norms, norm(e1 * t1 + c))
            end

            t2 = -dot(e2, c) / dot(e2, e2)
            if (0 < t2 < 1)
                push!(norms, norm(e2 * t2 + c))
            end

            t3 = -dot(e3, b) / dot(e3, e3)
            if (0 < t3 < 1)
                push!(norms, norm(e3 * t3 + b))
            end

            if !isempty(norms)
                push!(min_norms, minimum(norms))
            end
        else
            push!(min_norms, norm(E * t + c))
        end
    end

    return minimum(min_norms) - radius
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
        geo_cyl1_ell1, geo_cyl1_ell2, geo_cyl1_mid, cyl1_mid = make_ellipse_from_points(
            model, mesh_size,
            vertex1, axis, radius,
            major1_p1, major1_p2,
            geo_major1_p1, geo_major1_p2
        )
        push!(inner_curves[e1], (geo_cyl1_ell1, geo_cyl1_ell2, cyl1_mid, geo_cyl1_mid))
        geo_cyl2_ell1, geo_cyl2_ell2, geo_cyl2_mid, cyl2_mid = make_ellipse_from_points(
            model, mesh_size,
            vertex2, axis, radius,
            major2_p1, major2_p2,
            geo_major2_p1, geo_major2_p2
        )
        push!(inner_curves[e2], (geo_cyl2_ell1, geo_cyl2_ell2, cyl2_mid, geo_cyl2_mid))

        #make cross ellipse connection
        geo_con1_l1, geo_con1_l2, geo_con1_mid, con1_mid = make_line_or_ellipse(
            model, mesh_size,
            mid_vertex, axis, radius,
            minor1_p2, minor2_p1,
            geo_minor1_p2, geo_minor2_p1
        )
        geo_con2_l1, geo_con2_l2, geo_con2_mid, con2_mid = make_line_or_ellipse(
            model, mesh_size,
            mid_vertex, axis, radius,
            minor1_p1, minor2_p2,
            geo_minor1_p1, geo_minor2_p2
        )
        geo_con3_l1, geo_con3_l2, geo_con3_mid, con3_mid = make_line_or_ellipse(
            model, mesh_size,
            mid_vertex, axis, radius,
            major1_p2, major2_p1,
            geo_major1_p2, geo_major2_p1
        )
        geo_con4_l1, geo_con4_l2, geo_con4_mid, con4_mid = make_line_or_ellipse(
            model, mesh_size,
            mid_vertex, axis, radius,
            major1_p1, major2_p2,
            geo_major1_p1, geo_major2_p2
        )
        geo_con5_l1, geo_con5_l2, geo_con5_mid, con5_mid = make_line_or_ellipse(
            model, mesh_size,
            mid_vertex, axis, radius,
            cyl1_mid, cyl2_mid,
            geo_cyl1_mid, geo_cyl2_mid
        )

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

function make_inner_surfaces(model, mesh_size, vertices, all_inner_curves)
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

function make_anode(model, mesh_size, radius)
    o = model.geo.addPoint(0, 0, 0, mesh_size)
    z1 = model.geo.addPoint(0, 0, radius, mesh_size)
    z2 = model.geo.addPoint(0, 0, -radius, mesh_size)
    y1 = model.geo.addPoint(0, radius, 0, mesh_size)
    y2 = model.geo.addPoint(0, -radius, 0, mesh_size)
    x1 = model.geo.addPoint(radius, 0, 0, mesh_size)
    x2 = model.geo.addPoint(-radius, 0, 0, mesh_size)

    z1y1 = model.geo.addCircleArc(z1, o, y1)
    z1y2 = model.geo.addCircleArc(z1, o, y2)
    z1x1 = model.geo.addCircleArc(z1, o, x1)
    z1x2 = model.geo.addCircleArc(z1, o, x2)

    z2y1 = model.geo.addCircleArc(z2, o, y1)
    z2y2 = model.geo.addCircleArc(z2, o, y2)
    z2x1 = model.geo.addCircleArc(z2, o, x1)
    z2x2 = model.geo.addCircleArc(z2, o, x2)

    y1x1 = model.geo.addCircleArc(y1, o, x1)
    y1x2 = model.geo.addCircleArc(y1, o, x2)
    y2x1 = model.geo.addCircleArc(y2, o, x1)
    y2x2 = model.geo.addCircleArc(y2, o, x2)

    c1 = model.geo.addCurveLoop([z1y1, y1x1, z1x1], -1, true)
    s1 = model.geo.addSurfaceFilling([c1])

    c2 = model.geo.addCurveLoop([z1y2, y2x1, z1x1], -1, true)
    s2 = model.geo.addSurfaceFilling([c2])

    c3 = model.geo.addCurveLoop([z1x2, y1x2, z1y1], -1, true)
    s3 = model.geo.addSurfaceFilling([c3])

    c4 = model.geo.addCurveLoop([z1y2, y2x2, z1x2], -1, true)
    s4 = model.geo.addSurfaceFilling([c4])

    c5 = model.geo.addCurveLoop([z2y1, y1x1, z2x1], -1, true)
    s5 = model.geo.addSurfaceFilling([c5])

    c6 = model.geo.addCurveLoop([z2y2, y2x1, z2x1], -1, true)
    s6 = model.geo.addSurfaceFilling([c6])

    c7 = model.geo.addCurveLoop([z2x2, y1x2, z2y1], -1, true)
    s7 = model.geo.addSurfaceFilling([c7])

    c8 = model.geo.addCurveLoop([z2y2, y2x2, z2x2], -1, true)
    s8 = model.geo.addSurfaceFilling([c8])

    anode_surfaces = [s1, s2, s3, s4, s5, s6, s7, s8]

    return model.geo.addSurfaceLoop(anode_surfaces), anode_surfaces
end

function fib_points(r, ρ)
    cnt = floor(Int, ρ * 4 * π * r^2)
    points = []
    for i in 1:cnt
        ϕ = acos(1 - 2 * (i - 0.5) / cnt)
        θ = π * (1 + sqrt(5)) * i
        push!(points, [
            r * sin(ϕ) * cos(θ),
            r * sin(ϕ) * sin(θ),
            r * cos(ϕ)
        ])
    end
    return points
end

function make_mesh_res_points(model, size_min, size_max, wire_radius, anode_radius, inner_bound; scale=5.0, outer_shell_cnt=2, inner_shell_cnt=3, shell_density=5.0)
    size_func(r, radius) = (size_max - size_min) * (1 - exp(-scale * abs(r - radius))) / (1 - exp(-scale * (1 + wire_radius - inner_bound))) + size_min
    outer_radi = LinRange(1 + wire_radius, anode_radius, outer_shell_cnt + 2)[2:(end-1)]
    inner_radi = LinRange(0, inner_bound, inner_shell_cnt + 2)[2:(end-1)]

    geo_points = []

    for r in inner_radi
        points = fib_points(r, shell_density / r^(3 / 2))
        size = size_func(r, inner_bound)
        append!(geo_points, [model.geo.addPoint(p[1], p[2], p[3], size) for p in points])
    end

    for r in outer_radi
        points = fib_points(r, shell_density / r^(3 / 2))
        size = size_func(r, 1 + wire_radius)
        append!(geo_points, [model.geo.addPoint(p[1], p[2], p[3], size) for p in points])
    end

    return geo_points
end

function make_mesh(app_cnt, cathode_radius, cathode_resolution, anode_radius, anode_resolution, wire_radius)
    json_data = JSON3.read("cathode_data/appratures_$(app_cnt).json")
    edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
    vertices = Vector{Vector{Float64}}(json_data["vertices"])
    face_centers = Vector{Vector{Float64}}(json_data["points"])
    abstract_edges, adj_list = conectivity_analysis(vertices, edges)
    abstract_faces = find_abstract_faces(vertices, face_centers, adj_list)
    scaled_anode_radius = anode_radius / cathode_radius
    scaled_wire_radius = wire_radius / cathode_radius
    inner_bound = find_inner_bound(vertices, abstract_faces, scaled_wire_radius)

    X, Y, Z, I, J, K = Nothing, Nothing, Nothing, Int[], Int[], Int[]
    println("Making Mesh ...")
    try
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        model = gmsh.model
        model.add("Fusion")
        cathode_mesh_size = scaled_wire_radius * (2 * π / cathode_resolution)
        anode_mesh_size = scaled_anode_radius * (2 * π / anode_resolution)

        geo_vertices = [model.geo.addPoint(x, y, z, cathode_mesh_size) for (x, y, z) in vertices]
        #geo_edges = [model.geo.addLine(geo_vertices[i], geo_vertices[j]) for (i, j) in abstract_edges]

        inter_ells = make_intersection_ellipses(model, cathode_mesh_size, vertices, geo_vertices, adj_list, scaled_wire_radius)
        cathode_surfaces, outer_curves, inner_curves = make_edge_wires(
            model, cathode_mesh_size,
            vertices, geo_vertices,
            abstract_edges, adj_list,
            inter_ells, scaled_wire_radius)
        model.geo.synchronize()

        outer_surfaces = make_outer_surfaces(model, cathode_mesh_size, vertices, geo_vertices, outer_curves, scaled_wire_radius)
        inner_surfaces = make_inner_surfaces(model, cathode_mesh_size, vertices, inner_curves)
        model.geo.synchronize()

        all_cathode_surfaces = Int32[]
        append!(all_cathode_surfaces, flatten_tags(cathode_surfaces))
        append!(all_cathode_surfaces, flatten_tags(outer_surfaces))
        append!(all_cathode_surfaces, flatten_tags(inner_surfaces))

        cathode = model.geo.addSurfaceLoop(all_cathode_surfaces)
        model.geo.synchronize()

        anode, anode_surfaces = make_anode(model, anode_mesh_size, scaled_anode_radius)
        model.geo.synchronize()

        chamber = model.geo.addVolume([anode, -cathode])
        model.geo.synchronize()

        mesh_res_points = make_mesh_res_points(model, cathode_mesh_size, anode_mesh_size, scaled_wire_radius, scaled_anode_radius, inner_bound)
        model.geo.synchronize()

        cathode_physical = model.addPhysicalGroup(2, all_cathode_surfaces, 1)
        model.setPhysicalName(2, cathode_physical, "cathode")

        anode_physical = model.addPhysicalGroup(2, anode_surfaces, 2)
        model.setPhysicalName(2, anode_physical, "anode")

        chamber_physical = model.addPhysicalGroup(3, [chamber], 3)
        model.setPhysicalName(3, chamber_physical, "chamber")

        model.mesh.generate(3)
        gmsh.write("chamber_mesh.msh")
        println("Mesh Saved to: chamber_mesh.msh")

        gmsh.fltk.run()

        cathode_entities = gmsh.model.getEntitiesForPhysicalGroup(2, 1)
        all_triangle_nodes = Int[]
        for entity in cathode_entities
            _, _, elem_node_tags = gmsh.model.mesh.getElements(2, entity)
            if !isempty(elem_node_tags)
                append!(all_triangle_nodes, elem_node_tags[1])
            end
        end
        unique_node_tags = unique(all_triangle_nodes)
        all_nodes_tags, all_nodes_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
        all_nodes_dict = Dict{Int,Vector{Float64}}()
        for (tag, coords) in zip(all_nodes_tags,
            Iterators.partition(all_nodes_coords, 3))
            all_nodes_dict[tag] = collect(coords)
        end

        node_dict = Dict{Int,Vector{Float64}}()
        for tag in unique_node_tags
            if haskey(all_nodes_dict, tag)
                node_dict[tag] = all_nodes_dict[tag]
            end
        end

        node_tags = collect(keys(node_dict))
        node_idx = Dict(tag => i - 1 for (i, tag) in enumerate(node_tags))
        X = [node_dict[tag][1] for tag in node_tags] .* cathode_radius
        Y = [node_dict[tag][2] for tag in node_tags] .* cathode_radius
        Z = [node_dict[tag][3] for tag in node_tags] .* cathode_radius
        for entity in cathode_entities
            _, _, elem_node_tags = gmsh.model.mesh.getElements(2, entity)
            if !isempty(elem_node_tags)
                for (i, j, k) in Iterators.partition(elem_node_tags[1], 3)
                    # All nodes should exist now since we got them specifically
                    push!(I, node_idx[i])
                    push!(J, node_idx[j])
                    push!(K, node_idx[k])
                end
            end
        end
    finally
        gmsh.finalize()
    end

    return (x=X, y=Y, z=Z, i=I, j=J, k=K)
end

function solve_electric_feild(ΔV, scale)
    model = GmshDiscreteModel("chamber_mesh.msh")
    order = 2
    ReFE = ReferenceFE(lagrangian, Float64, order)
    V = TestFESpace(model, ReFE, dirichlet_tags=["cathode", "anode"])
    U = TrialFESpace(V, [-ΔV, 0.0])
    Ω = Triangulation(model)
    dΩ = Measure(Ω, 2 * order - 1)
    a(u, v) = ∫(∇(v) ⋅ ∇(u))dΩ
    l(v) = 0.0
    println("Building operator...")
    op = AffineFEOperator(a, l, U, V)
    println("Degrees of freedom: ", num_free_dofs(U))

    println("Solving for potential...")
    Φ = solve(op)
    println("Solve for potential complete.")

    return function (p::Vector{Float64})
        return Φ(Point(p ./ scale))
    end
end

app_cnt = 12
cathode_radius = 0.05
cathode_resolution = 8
anode_radius = 0.25
anode_resolution = 24
wire_radius = 0.0025
ΔV = 4e5
T = 293.15
md = 3.343e-27
e = 1.602e-16
kb = 1.380649e-23

plot_mesh_info = make_mesh(app_cnt, cathode_radius, cathode_resolution, anode_radius, anode_resolution, wire_radius)
