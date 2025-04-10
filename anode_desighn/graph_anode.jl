using LinearAlgebra
using Random
using PlotlyJS
using .Threads
using DataFrames
using CSV
using Combinatorics
import Base: ==, hash
using UUIDs
using JSON

struct Point
	id::UUID
	cords::Vector{Float64}
end

function ==(p1::Point, p2::Point)
	return p1.id == p2.id
end

function hash(p::Point, h::UInt)
	return hash(p.id, h::UInt)
end

function stereo_proj(points::Vector{Point}; ϵ = 1e-8)::Tuple{Vector{Point}, Vector{Point}}
	proj_points_up = []
	proj_points_down = []
	for p in points
		if abs(p.cords[3]-1) > ϵ
			push!(proj_points_down, Point(p.id, [p.cords[1], p.cords[2]] ./ (1 - p.cords[3])))
		end
		if abs(p.cords[3] + 1) > ϵ
			push!(proj_points_up, Point(p.id, [p.cords[1], p.cords[2]] ./ (1 + p.cords[3])))
		end
	end

	return proj_points_down, proj_points_up
end

struct Polygon
	points::Set{Point}
	edges::Set{Set{Point}}
end

function ==(p1::Polygon, p2::Polygon)
	return (p1.points == p2.points) && (p1.edges == p2.edges)
end

function hash(p::Polygon, h::UInt)
	return hash((p.points, p.edges), h)
end

function super_triangle(points::Vector{Point})::Polygon
	_, idx = findmax(point -> norm(point.cords), points)
	M = points[idx].cords
	R = [-1 -sqrt(3); sqrt(3) -1]
	p1 = Point(uuid4(), 2 .* M)
	p2 = Point(uuid4(), R * M)
	p3 = Point(uuid4(), R' * M)
	edges = Set([Set([p1, p2]), Set([p1, p3]), Set([p2, p3])])
	return Polygon(Set([p1, p2, p3]), edges)
end

function circumcircle(tri::Polygon)::Tuple{Vector{Float64}, Float64}
    @assert length(tri.points) == 3 
	a, b, c = Tuple([p.cords for p in tri.points])

    A = det([
        a[1] a[2] 1
        b[1] b[2] 1
        c[1] c[2] 1
    ])

    if abs(A) == 0
        return zeros(2), Inf
    end

    B = det([
        a[1] a[2] a[1]^2+a[2]^2
        b[1] b[2] b[1]^2+b[2]^2
        c[1] c[2] c[1]^2+c[2]^2
    ])

    Sx = (1/2)*det([
        a[1]^2+a[2]^2 a[2] 1
        b[1]^2+b[2]^2 b[2] 1
        c[1]^2+c[2]^2 c[2] 1
    ])
    
    Sy = (1/2)*det([
        a[1] a[1]^2+a[2]^2 1
        b[1] b[1]^2+b[2]^2 1
        c[1] c[1]^2+c[2]^2 1
    ])
	
    S = [Sx, Sy]
    
    return S./A , sqrt(B/A + (norm(S)/A)^2)
end

function merge(polys::Set{Polygon})::Polygon
	if length(polys) == 1
		return only(polys)
	end
	points = reduce(union, [poly.points for poly in polys])
	edges = reduce(union, [poly.edges for poly in polys])
	in_edges = Set{Set{Point}}()
	for p1 in polys
		for p2 in polys
			if (p1 != p2)
				union!(in_edges, intersect(p1.edges, p2.edges))
			end
		end
	end
	setdiff!(edges, in_edges)

	return Polygon(points, edges)
end

function triangulate(points::Vector{Point})::Set{Polygon}
	super_tri = super_triangle(points)
	triangulation = Set([super_tri])
	for p in points
		bad_tris = Set{Polygon}()
		for poly in triangulation
			center, radius = circumcircle(poly)
			if norm(center .- p.cords) <= radius
				push!(bad_tris, poly)
			end
		end

		hole = merge(bad_tris)

		for poly in bad_tris
			delete!(triangulation, poly)
		end

		for edge in hole.edges
			ep1, ep2 = Tuple(edge)
			new_points = Set([ep1, ep2, p])
			edges = Set([edge, Set([ep1, p]), Set([ep2, p])])
			push!(triangulation, Polygon(new_points, edges))
		end
	end

	for tri in triangulation
		if !isempty(intersect(tri.points, super_tri.points))
			delete!(triangulation, tri)
		end
	end

	return triangulation
end

function make_graph(tessalation::Set{Polygon})::Tuple{Vector{Vector{Float64}}, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}
	unique_points = reduce(union, [poly.points for poly in tessalation])
	unique_edges = reduce(union, [poly.edges for poly in tessalation])

	vertecies = [p.cords for p in unique_points]
    edges = []
	for e in unique_edges
        ep1 , ep2 = Tuple(e)
        push!(edges, Tuple(sort([ep1.cords, ep2.cords])))
    end

	return vertecies, edges
end

function plot_graph_2d(vertecies::Vector{Vector{Float64}}, edges::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
	v_mat = hcat(vertecies...)
	v_trace = scatter(
		x = v_mat[1, :],
		y = v_mat[2, :],
		mode = "markers",
		marker = attr(
			size = 5,
			color = "blue",
		),
	)

	traces = [v_trace]
	T = LinRange(0, 1, 100)
	for edge in edges
		ep1, ep2 = edge
		diff = ep1 .- ep2
		f(t) = ep2 .+ diff .* t
		E = hcat(f.(T)...)
		e_trace = scatter(
			x = E[1, :],
			y = E[2, :],
			mode = "line",
			line = attr(
				size = 3,
				color = "green",
			),
		)
		push!(traces, e_trace)
	end

	return traces
end

function inv_stero_point_down(p::Point)::Point
    cords = [2*p.cords[1], 2*p.cords[2], norm(p.cords)^2 - 1] ./ (1 + norm(p.cords)^2)
    return Point(p.id, cords)
end

function inv_stero_point_up(p::Point)::Point
    cords = [2*p.cords[1], 2*p.cords[2], 1 - norm(p.cords)^2] ./ (1 + norm(p.cords)^2)
    return Point(p.id, cords)
end

function inv_stereo(down::Set{Polygon}, up::Set{Polygon})::Set{Polygon}
    tessalation = Set{Polygon}()
    for poly in down
        points = Set{Point}()
        for p in poly.points
            push!(points, inv_stero_point_down(p))
        end
        edges = Set{Set{Point}}()
        for e in poly.edges
            ep1 , ep2 = Tuple(e)
            push!(edges, Set([inv_stero_point_down(ep1), inv_stero_point_down(ep2)]))
        end

        push!(tessalation, Polygon(points, edges))
    end

    for poly in up
        points = Set{Point}()
        for p in poly.points
            push!(points, inv_stero_point_up(p))
        end
        edges = Set{Set{Point}}()
        for e in poly.edges
            ep1 , ep2 = Tuple(e)
            push!(edges, Set([inv_stero_point_up(ep1), inv_stero_point_up(ep2)]))
        end

        push!(tessalation, Polygon(points, edges))
    end

    return tessalation
end

function plot_graph_3d(vertecies::Vector{Vector{Float64}}, edges::Vector{Tuple{Vector{Float64}, Vector{Float64}}})
	v_mat = hcat(vertecies...)
	v_trace = scatter3d(
		x = v_mat[1, :],
		y = v_mat[2, :],
        z = v_mat[3, :],
		mode = "markers",
		marker = attr(
			size = 3,
			color = "blue",
		),
	)

	traces = [v_trace]
	T = LinRange(0, 1, 100)
	for edge in edges
		ep1, ep2 = edge
		diff = ep1 .- ep2
		f(t) = ep2 .+ diff .* t
		E = hcat(f.(T)...)
		e_trace = scatter3d(
			x = E[1, :],
			y = E[2, :],
            z = E[3, :],
			mode = "lines",
			line = attr(
				size = 3,
				color = "green",
			),
		)
		push!(traces, e_trace)
	end

	return traces
end

function normal_vector(poly::Polygon)::Vector{Float64}
    p = sort([p.cords for p in poly.points])
    normal = cross(p[2] .- p[1], p[3] .- p[1])
    return normal ./ norm(normal)
end

function BFS!(poly::Polygon, merger::Set{Polygon}, visited::Set{Polygon}, tessalation::Set{Polygon}, ϵ::Float64)
    push!(merger, poly)
    normal = normal_vector(poly)
    other_polys = setdiff(tessalation, visited)
    for other_poly in other_polys
        if (other_poly != poly) && (!isempty(intersect(poly.edges, other_poly.edges)))
            push!(visited, other_poly)
            other_normal = normal_vector(other_poly)
            if (2/π)*acos(abs(dot(normal, other_normal))) < ϵ
                BFS!(other_poly, merger, visited, tessalation, ϵ)
            end
        end
    end
end

function prune(tessalation::Set{Polygon}; ϵ = 1e-2)::Set{Polygon}
    all_mergers = Set{Polygon}()
    mergers = []
    for poly in tessalation
        if !(poly ∈ all_mergers)
            merger = Set{Polygon}()
            visited = Set{Polygon}()
            BFS!(poly, merger, visited, tessalation, ϵ)

            push!(mergers, merger)
            union!(all_mergers, merger)

            if all_mergers == tessalation
                break
            end
        end
    end

    new_tessalation = Set{Polygon}()
    for merger in mergers
        push!(new_tessalation, merge(merger))
    end

    return new_tessalation
end

function make_cograph(tessalation::Set{Polygon})::Tuple{Vector{Vector{Float64}}, Vector{Tuple{Vector{Float64}, Vector{Float64}}}}
    centroids = Dict{Polygon, Vector{Float64}}()
    vertecies = []
    for poly in tessalation
        S = reduce(.+, [p.cords for p in poly.points])
        S ./= norm(S)
        push!(vertecies, S)
        centroids[poly] = S
    end

    unique_edges = Set{Set{Vector{Float64}}}()
    for poly in tessalation
        for other_poly in tessalation
            if (other_poly != poly) && (!isempty(intersect(poly.edges, other_poly.edges)))
                push!(unique_edges, Set([centroids[poly], centroids[other_poly]]))
            end
        end
    end

    edges = [Tuple(edge) for edge in unique_edges]

    return vertecies, edges
end


filename = "anode_data/appratures_8.json"
data = JSON.parsefile(filename)
point_mat = hcat(data["points"]...)
points = [Point(uuid4(), col) for col in eachcol(point_mat)]
stereos = stereo_proj(points)
triangulations = triangulate.(stereos)
tessalation = inv_stereo(triangulations...)
pruned_tessalation = prune(tessalation)
v, e = make_graph(tessalation)

traces = plot_graph_3d(v,e)
plot(traces)





