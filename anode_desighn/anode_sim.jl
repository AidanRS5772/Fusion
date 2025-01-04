using Base: Process
using LinearAlgebra
using JSON
using PlotlyJS
using StaticArrays
using JacobiElliptic
using SpecialFunctions
using BenchmarkTools
using DifferentialEquations
using Cubature
using .Threads
using Distributions


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

function plot_path(sol)
    return scatter3d(
        x=sol[1, :],
        y=sol[2, :],
        z=sol[3, :],
        mode="lines",
        line=attr(
            size=2,
            color="red"
        )
    )
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

#Scaled by one 1/ε_0
function capacitence(edges, L, R, ω; rtol=1e-4)
    @inline function V_int(ρ, ζ, z)
        val = (ζ - z)^2 + (ρ + ω)^2
        return ellipk(clamp(4 * ρ * ω / val, 0.0, 1.0 - eps())) / sqrt(val)
    end

    global u = MVector{3,Float64}(0.0, 0.0, 0.0)
    V1 = 0.0
    V2 = 0.0

    V1_int(x::Vector{Float64}) = V_int(ω, x[1], x[2])

    for (m, z_vec) in edges
        l = norm(z_vec)
        function V2_int(x::Vector{Float64})
            z, θ, ϕ = x

            u[1] = cos(θ) * sin(ϕ)
            u[2] = sin(θ) * sin(ϕ)
            u[3] = cos(ϕ)
            u .*= R

            ζ = dot(z_vec, u .- m) / l
            ρ = norm(u .- m .- ζ .* z_vec ./ l)
            return V_int(ρ, ζ, z) * sin(ϕ)
        end

        V1 += hcubature(V1_int, (-l / 2, -l / 2), (l / 2, l / 2), reltol=rtol)[1]
        V2 += hcubature(V2_int, (-l / 2, 0.0, 0.0), (l / 2, 2 * π, π), reltol=rtol)[1]
    end

    V1 /= 19.7392088 * L^2
    V2 /= 11.62735376 * R * L

    return 1 / abs(V1 - V2)
end

function make_initial(u, hb, R, T)
    r = (u[1] * (R^3 - hb^3) + hb^3)^(1 / 3)
    sϕ = 2 * sqrt(u[2]) * sqrt(1 - u[2])
    θ = 2 * π * u[3]
    v = (90.87627864 * sqrt(T)) * erfinv.(1 .- 2 .* u[4:6])
    return (MVector{3,Float64}(
            r * cos(θ) * sϕ,
            r * sin(θ) * sϕ,
            r * (1 - 2 * u[2])
        ),
        MVector{3,Float64}(v...))
end

function lorenz!(ddu, du, u, p, t)
    edges, ω, scale = p
    ddu .= 0.0
    for (m, e) in edges
        z_vec = MVector(e)
        l = norm(z_vec)
        z_vec ./= l
        d = u .- m
        ζ = dot(z_vec, d)
        r_vec = d .- ζ .* z_vec
        ρ = norm(r_vec)
        r_vec ./= ρ

        a = 4 * ρ * ω
        b = (ρ + ω)^2
        zp, zm = ζ + l / 2, ζ - l / 2
        cp, cm = zp^2 + b, zm^2 + b
        Kp, Km = ellipk(a / cp), ellipk(a / cm)
        ds = (ρ - ω) / (ρ + ω)
        scp, scm = sqrt(cp), sqrt(cm)

        Ez = Kp / scp - Km / scm
        Er = (zp / scp) * (ds * Pi(a / b, a / cp) + Kp) - (zm / scm) * (ds * Pi(a / b, a / cm) + Km)
        Er /= 2 * ρ

        ddu .+= Er .* r_vec - Ez .* z_vec
    end
    ddu .*= -scale
end

function intersect(p, m, z, ω)
    v1 = m .+ z ./ 2
    v2 = m .- z ./ 2
    n = norm(z)
    r = norm(cross(p .- v1, p .- v2)) / n
    if r <= ω
        t = dot(p - m, z) / n
        if -n / 2 <= t <= n / 2
            return true
        end
    end
    return false
end

function find_path(edges, hb, lb, max_orbit, ω, scale, u0, v0; dt=1e-9)
    orbit_cnt = 0
    n1 = norm(u0)
    n2 = n1
    function condition(val, _, _)
        _, u = val.x
        n = norm(u)
        if (n < n1) && (n2 < n1) && (n1 > hb)
            orbit_cnt += 1
            if orbit_cnt >= max_orbit
                return true
            end
        end

        if min(n, n1) < hb || max(n, n1) > lb
            for (m, z) in edges
                if intersect(u, m, z, ω)
                    return true
                end
            end
        end

        n1, n2 = n, n1

        return false
    end

    cb = DiscreteCallback(condition, terminate!)
    prob = SecondOrderODEProblem(lorenz!, v0, u0, (0.0, 1e-2), (edges, ω, scale))
    _ = solve(prob, KahanLi8(), dt=dt, callback=cb, save_everystep=false)

    return orbit_cnt
end

function process_sol(sol)
    position = []
    for val in sol.u
        v, u = val.x
        push!(position, Vector(u))
    end

    return hcat(position...)
end

function do_analysis(app_cnt, N_samples, max_orbit, r, R, ω, V, T)
    data = open("anode_data/appratures_$(app_cnt).json", "r") do file
        JSON.parse(file)
    end

    edges, L, lb, hb = make_edges(data["edges"], r, ω)

    C = capacitence(edges, L, R, ω)
    data["capacitence"] = C
    s = 2427551.451 * V * C / L

    seeds = rand(6 * N_samples)

    orbits = Vector{Float64}(undef, N_samples)
    initials = Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, N_samples)

    println("Starting Sampling for $(app_cnt) Appratures...")
    @threads for i in 1:N_samples
        u0, v0 = make_initial(seeds[6*i-5:6*i], hb, R, T)
        initials[i] = (Vector(u0), Vector(v0))
        orbits[i] = find_path(edges, hb, lb, max_orbit, ω, s, u0, v0)
    end
    println("Done.")

    data["raw_orbit_counts"] = sort(orbits, rev=true)

    eternal_initials = []
    filtered_orbits = []
    for i in eachindex(orbits)
        if orbits[i] == max_orbit
            push!(eternal_initials, initials[i])
        else
            push!(filtered_orbits, orbits[i])
        end
    end
    filtered_orbits = convert(Vector{Float64}, filtered_orbits)
    eternal_orbit_cnt = length(eternal_initials)

    gamma_fit = fit_mle(Gamma, filtered_orbits .+ 1.0)
    adj_eternal_prop = (length(eternal_initials) / N_samples) - ccdf(gamma_fit, max_orbit .+ 1.0)

    mean_val = mean(filtered_orbits)
    median_val = median(filtered_orbits)
    mode_val = mode(filtered_orbits)
    std_val = std(filtered_orbits)
    skewness_val = skewness(filtered_orbits)
    kurtosis_val = kurtosis(filtered_orbits)

    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    q_values = quantile(filtered_orbits, quantiles)

    α = shape(gamma_fit)
    θ = scale(gamma_fit)

    h = histogram(
        x=orbits,
        opacity=0.5,
        marker_color="blue",
        histnorm="probability",
        name="Data",
        xbins=Dict(
            "start" => -0.5,  # Start half-way between 0 and first value
            "end" => max_orbit + 0.5,  # End half-way after last possible value
            "size" => 1.0    # Bin size of 1 to capture each integer value
        )
    )

    x_range = range(0, max_orbit, length=100)
    y_fitted = pdf.(gamma_fit, x_range .+ 1.0)

    # Create vertical lines as shapes in the layout
    shapes = [
        vline(mean_val, line_dash="dash", line_color="red"),
        vline(median_val, line_dash="dash", line_color="red"),
        vline(mode_val, line_dash="dash", line_color="red")
    ]

    analysis_plot = plot([
            h,
            scatter(x=x_range, y=y_fitted, mode="lines",
                name="Fitted Gamma", line_color="red")
        ],
        Layout(
            title="Orbit Distribution Analysis",
            xaxis_title="Number of Orbits",
            yaxis_title="Probability Density",
            showlegend=true,
            shapes=shapes,
            # Add annotations for the lines
            annotations=[
                attr(
                    x=mean_val,
                    y=0.9,
                    text="Mean = $(round(mean_val, digits = 2))",
                    showarrow=false,
                    yref="paper",
                    font_color="red"
                ),
                attr(
                    x=median_val,
                    y=0.95,
                    text="Median = $(round(median_val, digits = 2))",
                    showarrow=false,
                    yref="paper",
                    font_color="red"
                ),
                attr(
                    x=mode_val,
                    y=1.0,
                    text="Mode = $(round(mode_val, digits = 2))",
                    showarrow=false,
                    yref="paper",
                    font_color="red"
                )
            ]
        )
    )

    # Print comprehensive analysis
    println("\nAnalysis Results:")

    println("Sample Size: ", length(filtered_orbits))
    println("Number of Eternal Orbits: ", eternal_orbit_cnt, " (",
        100 * eternal_orbit_cnt / N_samples, "%)")
    data["eternal_initials"] = eternal_initials

    println("\nBasic Statistics:")

    println("Mean: ", mean_val)
    data["mean"] = mean_val

    println("Median: ", median_val)
    data["median"] = median_val

    println("Mode: ", mode_val)
    data["mode"] = mode_val

    println("Standard Deviation: ", std_val)
    data["std"] = std_val

    println("Skewness: ", skewness_val)
    data["skew"] = skewness_val

    println("Kurtosis: ", kurtosis_val)
    data["Kurtosis"] = kurtosis_val

    println("\nQuantiles:")
    for (q, v) in zip(quantiles, q_values)
        println(q * 100, "th percentile: ", v)
    end
    data["quantiles"] = collect(zip(quantiles, q_values))

    println("\nGamma Distribution Parameters:")

    println("Shape (α): ", α)
    data["gamma_shape"] = α

    println("Scale (θ): ", θ)
    data["gamma_scale"] = θ

    println("\nTail Analysis:")
    println("Probability of exceeding ", max_orbit, " orbits with Gamma fit: ",
        ccdf(gamma_fit, max_orbit .+ 1))
    println("Adjusted proportion of eternal orbits: ", adj_eternal_prop)
    data["adj_prop_eternal"] = adj_eternal_prop

    println("")

    open("anode_data/appratures_$(app_cnt).json", "w") do f
        JSON.print(f, data)
    end

    savefig(analysis_plot, "anode_data_plots/appratures_$(app_cnt).html")
    run(`open anode_data_plots/appratures_$(app_cnt).html`)
end


r, R, ω, V, T = 0.05, 0.25, 0.001, 1e5, 3e2
N_samples = 5000
max_orbit = 50

for i in 6:2:362
    do_analysis(i, N_samples, max_orbit, r, R, ω, V, T)
end
