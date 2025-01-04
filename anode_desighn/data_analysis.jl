using PlotlyJS
using JSON
using LinearAlgebra
using StaticArrays

function cube_layout(lim)
    return Layout(
        scene=attr(
            aspectmode="cube",
            xaxis=attr(range=[-lim, lim]),
            yaxis=attr(range=[-lim, lim]),
            zaxis=attr(range=[-lim, lim])
        )
    )
end

function plot_cylinder(radius, length, direction, midpoint; color="grey", opacity=0.7, res=5)
    direction = direction / norm(direction)
    temp = direction ≈ [1.0, 0.0, 0.0] ? [0.0, 1.0, 0.0] : [1.0, 0.0, 0.0]
    orthogonal1 = normalize(cross(direction, temp))
    orthogonal2 = normalize(cross(direction, orthogonal1))

    n_theta, n_z = res, res
    theta = range(0, 2π, length=n_theta)
    z = range(-length / 2, length / 2, length=n_z)

    Θ = repeat(theta, 1, n_z)
    Z_local = repeat(z', n_theta, 1)

    X_local = radius .* cos.(Θ)
    Y_local = radius .* sin.(Θ)

    X = midpoint[1] .+ X_local .* orthogonal1[1] .+ Y_local .* orthogonal2[1] .+ Z_local .* direction[1]
    Y = midpoint[2] .+ X_local .* orthogonal1[2] .+ Y_local .* orthogonal2[2] .+ Z_local .* direction[2]
    Z = midpoint[3] .+ X_local .* orthogonal1[3] .+ Y_local .* orthogonal2[3] .+ Z_local .* direction[3]

    return PlotlyJS.surface(
        x=X,
        y=Y,
        z=Z,
        colorscale=[[0, color], [1, color]],
        opacity=opacity,
        showscale=false
    )
end

function plot_anode(edges, ω)
    traces = [plot_cylinder(ω, norm(edges[1][2]), edges[1][2], edges[1][1])]
    for e in edges[2:end]
        trace = plot_cylinder(ω, norm(e[2]), e[2], e[1])
        push!(traces, trace)
    end

    return traces
end

function make_edges(data_edges, r, ω)
    edges = []
    L = 0.0
    lb = Inf
    hb = 0.0
    for e in data_edges
        e1, e2 = e[1] .* r, e[2] .* r
        m = (e1 .+ e2) ./ 2
        z = e1 .- e2
        L += norm(z)
        if norm(m) < lb
            lb = norm(m)
        end
        if norm(e1) > hb
            hb = norm(e1)
        end
        if norm(e2) > hb
            hb = norm(e2)
        end
        push!(edges, (SVector{3,Float64}(m...), SVector{3,Float64}(z...)))
    end

    return SVector{length(edges),Tuple{SVector{3,Float64},SVector{3,Float64}}}(edges...), L, lb - ω, hb + ω
end

data = open("anode_data/appratures_6.json", "r") do file
    JSON.parse(file)
end

edges, _, _, _ = make_edges(data["edges"], 0.05, 0.001)
eternal_initials = data["eternal_initials"]

pos = []
vel = []
for (p, v) in eternal_initials
    push!(pos, p)
    push!(vel, v)
end

pos = hcat(pos...)
vel = hcat(vel...)

pos_trace = scatter3d(
    x=pos[1, :],
    y=pos[2, :],
    z=pos[3, :],
    mode="markers",
    marker=attr(
        size=3,
        color="red"
    )
)

plot([plot_anode(edges, 0.001)..., pos_trace], cube_layout(0.5))
