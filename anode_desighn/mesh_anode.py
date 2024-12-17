import numpy as np
import trimesh
import json
import plotly.graph_objects as go
import open3d as o3d

def create_cylinder(p1, p2, radius, sections):
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length == 0:
        raise ValueError("p1 and p2 cannot be the same point.")
    direction = vec / length

    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=sections
    )

    z_axis = np.array([0, 0, 1])
    rotation = trimesh.geometry.align_vectors(z_axis, direction)
    cylinder.apply_transform(rotation)
    cylinder.apply_translation((p1 + p2) / 2.0)

    return cylinder

def create_sphere(p, radius, subdivisions):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(p)
    return sphere

def plot_mesh(mesh, is_o3d=False):
    # Get vertices and faces depending on mesh type
    if is_o3d:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
    else:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

    # Create plotly figure
    fig = go.Figure(data=[
        go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                 i=faces[:,0], j=faces[:,1], k=faces[:,2],
                 color='grey', opacity=0.5)
    ])
    fig.show()

def trimesh_to_open3d(trimesh_mesh):
    # Ensure faces are the correct data type
    vertices = np.array(trimesh_mesh.vertices, dtype=np.float64)
    faces = np.array(trimesh_mesh.faces, dtype=np.int32)

    # Create mesh
    o3d_mesh = o3d.geometry.TriangleMesh()

    # Convert vertices first
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Convert faces in smaller chunks to avoid memory issues
    chunk_size = 1000
    for i in range(0, len(faces), chunk_size):
        chunk = faces[i:i + chunk_size]
        if i == 0:
            o3d_mesh.triangles = o3d.utility.Vector3iVector(chunk)
        else:
            o3d_mesh.triangles.extend(o3d.utility.Vector3iVector(chunk))

    return o3d_mesh

def clean_mesh(o3d_mesh):
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_non_manifold_edges()
    o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh.orient_triangles()
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()

    return o3d_mesh

for i in range(6,364,2):
    with open(f'anode_data/appratures_{i}.json', 'r') as file:
        data = json.load(file)

    print(f"{i} Appratures:")
    vertices = [np.array(p) for p in data['vertices']]
    edges = [(np.array(pair[0]), np.array(pair[1])) for pair in data['edges']]

    radius = .02*1
    cylinders = [create_cylinder(e[0], e[1], radius, sections=6) for e in edges]
    spheres = [create_sphere(v, radius, subdivisions=1) for v in vertices]
    mesh = trimesh.boolean.union(cylinders + spheres, engine='manifold')

    print(f"Original number of faces: {len(mesh.faces)}")
    target_faces = max(1500, len(mesh.faces) // int(np.ceil(np.log(len(mesh.faces)))))
    o3d_mesh = trimesh_to_open3d(mesh)
    decimated_mesh = o3d_mesh.simplify_quadric_decimation(target_faces)
    decimated_mesh = clean_mesh(decimated_mesh)
    print(f"Decimated number of faces: {len(np.asarray(decimated_mesh.triangles))}")
    try:
        # Try saving with Open3D
        o3d.io.write_triangle_mesh(f"anode_meshes/appratures_{i}.stl", decimated_mesh)
    except Exception as e:
        print(f"Open3D save failed: {e}")
        print("Attempting to save with trimesh...")
        # Convert to trimesh and save
        vertices = np.asarray(decimated_mesh.vertices)
        faces = np.asarray(decimated_mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.export(f"anode_meshes/appratures_{i}.stl")
