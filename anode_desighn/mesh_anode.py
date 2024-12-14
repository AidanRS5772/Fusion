import numpy as np
import trimesh
import json
import plotly.graph_objects as go

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
    
    midpoint = (p1 + p2) / 2.0
    cylinder.apply_translation(midpoint)
    
    return cylinder

def create_sphere(p, radius, subdivisions):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.apply_translation(p)
    return sphere

def plot_mesh_with_plotly(mesh):
    # Extract the mesh data
    vertices = mesh.vertices
    faces = mesh.faces

    # Create the Plotly mesh
    mesh_plotly = go.Mesh3d(
        x=vertices[:,0],
        y=vertices[:,1],
        z=vertices[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        color='grey',
        opacity=0.5
    )

    # Define the layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True),
            aspectmode='data'
        ),
        title="3D Mesh Visualization with Plotly"
    )

    # Create the figure and display it
    fig = go.Figure(data=[mesh_plotly], layout=layout)
    fig.show()

with open('anode_data/appratures_300.json', 'r') as file:
    data = json.load(file) 

vertices = [np.array(p) for p in data['vertices']]
edges = [(np.array(pair[0]), np.array(pair[1])) for pair in data['edges']]

radius = .002*10
sections = 10
subdivisions = 4
cylinders = [create_cylinder(e[0], e[1], radius, sections) for e in edges]
spheres = [create_sphere(v, radius, subdivisions) for v in vertices]
mesh = trimesh.boolean.union(cylinders + spheres, engine='manifold')

plot_mesh_with_plotly(mesh)
