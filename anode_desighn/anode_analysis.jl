using JSON
using GeometryBasics
using MeshIO
using FileIO
using StructArrays
using PlotlyJS
using Clustering

function scale_mesh(mesh, scale_factor)
    verts = coordinates(mesh)

    new_vertices = StructVector{typeof(verts[1])}((
        position=[Point{3,Float32}(v.position .* scale_factor) for v in verts],
        normals=verts.normals
    ))

    return GeometryBasics.Mesh(new_vertices, faces(mesh))
end

function plot_mesh(mesh)
    # For each triangle, get its three vertices
    x = Float64[]
    y = Float64[]
    z = Float64[]
    i = Int[]
    j = Int[]
    k = Int[]

    # For each triangle in the mesh
    for (idx, triangle) in enumerate(mesh)
        # Get the three vertices
        v1, v2, v3 = triangle[1], triangle[2], triangle[3]

        # Add vertex coordinates
        push!(x, v1[1], v2[1], v3[1])
        push!(y, v1[2], v2[2], v3[2])
        push!(z, v1[3], v2[3], v3[3])

        # Add face indices (3 vertices per triangle)
        base = (idx - 1) * 3  # each triangle starts 3 vertices after the previous
        push!(i, base)
        push!(j, base + 1)
        push!(k, base + 2)
    end

    # Create mesh trace
    mesh_trace = mesh3d(
        x=x,
        y=y,
        z=z,
        i=i,
        j=j,
        k=k,
        opacity=0.5,
        color="grey"
    )

    return mesh_trace
end

function plot_vectors(M; scale=0.02)
    X = M[1, :]
    Y = M[2, :]
    Z = M[3, :]
    U = M[4, :]
    V = M[5, :]
    W = M[6, :]

    # Compute magnitudes
    magnitudes = sqrt.(U .^ 2 .+ V .^ 2 .+ W .^ 2)
    min_mag = minimum(magnitudes)
    max_mag = maximum(magnitudes)

    # Normalize directions to unit vectors
    U_norm = U ./ magnitudes
    V_norm = V ./ magnitudes
    W_norm = W ./ magnitudes

    # Choose a fixed length L for all vectors
    L = scale  # Adjust as needed for visibility

    # Normalize magnitudes to [0,1] for color mapping
    normalized_mags = (magnitudes .- min_mag) ./ (max_mag - min_mag)

    # Define a helper function to map a normalized value in [0,1] to a color between red and blue.
    function colormap_red_blue(val)
        # val in [0,1]; at 0 -> red, at 1 -> blue
        r = 1.0 - val
        g = 0.0
        b = val
        hr = string(round(Int, r * 255), base=16, pad=2)
        hg = string(round(Int, g * 255), base=16, pad=2)
        hb = string(round(Int, b * 255), base=16, pad=2)
        "#" * uppercase(hr) * uppercase(hg) * uppercase(hb)
    end

    # Create a list of traces, one per vector
    traces = []
    for i in 1:size(M, 2)
        # Start point
        x_start = X[i]
        y_start = Y[i]
        z_start = Z[i]

        # End point = start + L*(unit direction)
        x_end = x_start + L * U_norm[i]
        y_end = y_start + L * V_norm[i]
        z_end = z_start + L * W_norm[i]

        # Color based on normalized magnitude
        c = colormap_red_blue(normalized_mags[i])

        # Line segment for this vector
        push!(traces, scatter3d(
            x=[x_start, x_end],
            y=[y_start, y_end],
            z=[z_start, z_end],
            mode="lines",
            line=attr(color=c, width=2),
            name="Vector $i"
        ))

        # Add a small cone at the tip to indicate direction
        # Use a tiny sizeref so the cone is small, adjust as needed
        push!(traces, cone(
            x=[x_end],
            y=[y_end],
            z=[z_end],
            u=[U_norm[i]],
            v=[V_norm[i]],
            w=[W_norm[i]],
            sizemode="absolute",
            sizeref=L * 0.3, # size of the cone relative to vector length, adjust as needed
            anchor="tip",
            colorscale=[[0, c], [1, c]], # solid color
            cmin=0, cmax=1,
            showscale=false,
            name="Tip $i"
        ))
    end

    return traces
end

app_cnt = 12
data = JSON.parsefile("anode_data/appratures_$(app_cnt).json")
mesh = scale_mesh(load("anode_meshes/appratures_$(app_cnt).stl"), 0.05)

eternals = hcat(data["eternal_initials"]...)
clusters = dbscan(eternals, 0.05)
display(clusters.counts)
