using JSON3
using Gmsh
using LinearAlgebra

function make_mesh(app_cnt, cathode_radius, anode_radius, wire_radius)
    json_data = JSON3.read("cathode_geometry/app_$(app_cnt).json")
    edges = Vector{Vector{Vector{Float64}}}(json_data["edges"])
    vertices = Vector{Vector{Float64}}(json_data["vertices"])

    scaled_anode_radius = anode_radius / cathode_radius
    scaled_wire_radius = wire_radius / cathode_radius

    println("scaled anode_radius: ", scaled_anode_radius)
    println("scaled wire_radius: ", scaled_wire_radius)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    model_name = "FEM_Mesh"
    gmsh.model.add(model_name)

    try
        objects = []
        for edge in edges
            diff = edge[2] - edge[1]
            start = edge[1]
            push!(objects, gmsh.model.occ.addCylinder(start[1], start[2], start[3], diff[1], diff[2], diff[3], scaled_wire_radius))
        end

        for vertex in vertices
            push!(objects, gmsh.model.occ.addSphere(vertex[1], vertex[2], vertex[3], scaled_wire_radius))
        end

        gmsh.model.occ.synchronize()

        cathode = objects[1]
        for object in objects[2:end]
            result, _ = gmsh.model.occ.fuse([(3, cathode)], [(3, object)])
            cathode = result[1][2]
            gmsh.model.occ.synchronize()
        end

        anode = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, scaled_anode_radius)
        gmsh.model.occ.synchronize()

        result, _ = gmsh.model.occ.cut([(3, anode)], [(3, cathode)])
        gmsh.model.occ.synchronize()
        chamber = result[1][2]
        chamber_surfaces = gmsh.model.getBoundary([(3, chamber)])

        boundary_areas = []
        for (dim, tag) in chamber_surfaces
            # Get surface area
            mass = gmsh.model.occ.getMass(dim, abs(tag))
            push!(boundary_areas, (mass, tag))
        end

        # Sort surfaces by area (largest first)
        sort!(boundary_areas, by=x -> x[1], rev=true)

        # The largest surface should be the anode (outer sphere)
        anode_boundary = [boundary_areas[1][2]]

        # All other surfaces belong to the cathode
        cathode_boundary = [item[2] for item in boundary_areas[2:end]]

        println("Anode boundary: ", anode_boundary, " with area: ", boundary_areas[1][1])
        println("Cathode boundary surfaces: ", length(cathode_boundary))

        # Debug - print areas of all surfaces to verify
        println("Surface areas:")
        for (area, tag) in boundary_areas
            println("Surface ", tag, ": ", area)
        end

        anode_tag = gmsh.model.addPhysicalGroup(2, anode_boundary)  # 2 is for surfaces
        cathode_tag = gmsh.model.addPhysicalGroup(2, cathode_boundary)
        chamber_tag = gmsh.model.addPhysicalGroup(3, [chamber])

        gmsh.model.setPhysicalName(2, anode_tag, "anode")
        gmsh.model.setPhysicalName(2, cathode_tag, "cathode")
        gmsh.model.setPhysicalName(3, chamber_tag, "chamber")

        # --- Mesh size control ---
        # 1. Set base sizes
        fine_mesh_size = scaled_wire_radius * 0.5  # Fine mesh near cathode
        coarse_mesh_size = scaled_anode_radius * 0.2  # Coarse mesh near anode

        # 2. Set global mesh parameters first
        gmsh.option.setNumber("Mesh.MeshSizeMin", fine_mesh_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", coarse_mesh_size)

        # 3. Create fields for advanced mesh control

        # Create distance field from ALL cathode surfaces
        cathode_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(cathode_field, "FacesList", cathode_boundary)

        # Create a threshold field to handle the size transition
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", cathode_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", fine_mesh_size)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", coarse_mesh_size)

        # Adjust these parameters to control the rate of transition
        # DistMin is where the fine mesh ends
        # DistMax is where the coarse mesh begins
        transition_start = scaled_wire_radius * 2  # Distance from cathode where gradation begins
        transition_end = scaled_anode_radius * 0.3  # Distance from cathode where max element size is reached

        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", transition_start)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", transition_end)

        # Add direct node size control for cathode points
        # This ensures the cathode surfaces have the fine mesh size
        for surface_tag in cathode_boundary
            # Get vertices of the cathode surface
            cathode_vertices = gmsh.model.getBoundary([(2, surface_tag)], false, false, true)
            for point in cathode_vertices
                # Force minimum mesh size at cathode vertices
                gmsh.model.mesh.setSize([point], fine_mesh_size)
            end
        end

        # Apply min of all fields
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])

        # Use the field as background mesh size field
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

        # Additional mesh quality settings
        gmsh.option.setNumber("Mesh.MinimumCirclePoints", 8)  # Increased for better circle approximation
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D meshes
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Optimize the final mesh
        gmsh.option.setNumber("Mesh.SmoothRatio", 1.5)  # Control gradation smoothness (reduced for better transition)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)  # Extend mesh size from boundary
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)  # Consider point sizes
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)  # Consider curvature for sizing

        # Add some debugging output
        println("Mesh sizing field created with transition from ", fine_mesh_size, " to ", coarse_mesh_size)
        println("Transition starts at distance ", transition_start, " and ends at ", transition_end)

        # Generate mesh
        gmsh.model.mesh.generate(2)  # Generate 3D mesh

        # Save mesh
        # gmsh.write("app_$(app_cnt)_mesh.msh")

        # Visualization
        gmsh.fltk.run()
    catch e
        println("Error: ", e)
    finally
        gmsh.finalize()
    end
end

make_mesh(42, 0.05, 0.25, 0.001)
