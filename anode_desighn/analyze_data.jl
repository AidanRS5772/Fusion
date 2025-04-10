using JSON
using PlotlyJS
using Statistics
using GeometryBasics
using MeshIO
using FileIO
using GLMakie
using LinearAlgebra
using KernelDensity

function plot_stl(file_path::String)
    # Read the STL file
    mesh = load(file_path)

    # Create a figure
    fig = Figure()

    # Create a 3D axis
    ax = Axis3(fig[1, 1],
        aspect=:data,
        xlabel="X",
        ylabel="Y",
        zlabel="Z"
    )

    # Plot the mesh
    mesh!(ax, mesh,
        color=:blue,
        transparency=false,
        shading=true
    )

    # Display the figure
    display(fig)

    return fig
end

function get_data(data_reqs, range)
    all_data = Dict(k => [] for k in data_reqs)
    for app_cnt in range
        anode_data = open("anode_data/appratures_$(app_cnt).json", "r") do file
            JSON.parse(file)
        end

        for data_req in data_reqs
            try
                push!(all_data[data_req], anode_data[data_req])
            catch
                println("Bad Key in Data request.")
                return
            end
        end
    end

    return all_data
end

function make_data_plots(data_reqs, l, h)
    range = l:2:h
    data = get_data(data_reqs, range)
    plots = []
    for data_req in data_reqs
        avg = mean(data[data_req])
        println("Average $(data_req): $(avg)")
        p = PlotlyJS.plot([
                PlotlyJS.scatter(
                    x=collect(range),
                    y=data[data_req],
                    mode="markers+lines",
                    name="Data"
                ),
                PlotlyJS.scatter(
                    x=[l, h],
                    y=[avg, avg],
                    mode="lines",
                    line=attr(
                        dash="dash",
                        color="red"
                    ),
                    name="Average")
            ],
            PlotlyJS.Layout(
                title="$(data_req) by Apprature Count",
                xaxis_title="Apprature Count",
                yaxis_title=data_req,
                showlegend=true,
            )
        )

        push!(plots, p)
    end

    return hcat(plots...)
end

function make_density_plot_1D(data, bandwidth=0.01)
    K = kde(data)
    D = []
    L = minimum(data):bandwidth:maximum(data)
    for i in 1:(length(L)-1)
        c = count(x -> L[i] <= x < L[i+1], data)
        push!(D, c)
    end

    D ./= sum(D) * bandwidth

    return PlotlyJS.plot([
        PlotlyJS.scatter(
            x=collect(L)[1:end-1],
            y=D,
            mode="markers",
            name="Data Density",
        ),
        PlotlyJS.scatter(
            x=collect(K.x),
            y=K.density,
            mode="lines",
            name="KDE",
        )
    ]
    )
end

# data = get_data(["eternal_initials"], 42:42);
# eternal_initials = data["eternal_initials"][1];
# positions = [initial[1] for initial in eternal_initials]
# velocities = [initial[2] for initial in eternal_initials]

# radius = [norm(pos) for pos in positions]
# # θ = [atan(pos[2], pos[1]) for pos in positions]
# # Φ = [acos(pos[3] / radius[i]) for (i, pos) in enumerate(positions)]

# make_density_plot_1D(radius)

plot_stl("anode_meshes/appratures_42.stl")
