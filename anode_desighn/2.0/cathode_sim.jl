using Core: ScalarIndex
using Base: Enumerate
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
        chamber = result[1][2]  # Set maximum element size

        chamber_surfaces = gmsh.model.getBoundary([(3, chamber)])
        _, idx = findmin([norm(gmsh.model.occ.getCenterOfMass(dim, abs(tag))) for (dim, tag) in chamber_surfaces])

        anode_boundary = [chamber_surfaces[idx][2]]
        cathode_boundary = [surface[2] for (i, surface) in enumerate(chamber_surfaces) if i != idx]
        anode_tag = gmsh.model.addPhysicalGroup(2, anode_boundary)  # 2 is for surfaces
        cathode_tag = gmsh.model.addPhysicalGroup(2, cathode_boundary)
        chamber_tag = gmsh.model.addPhysicalGroup(3, [chamber])

        gmsh.model.setPhysicalName(2, anode_tag, "anode")
        gmsh.model.setPhysicalName(2, cathode_tag, "cathode")
        gmsh.model.setPhysicalName(3, chamber_tag, "chamber")

        gmsh.model.mesh.generate(2)
        gmsh.fltk.run()

    catch e
        println("Error: ", e)
    finally
        gmsh.finalize()
    end
end

make_mesh(42, 0.05, 0.25, 0.001)
