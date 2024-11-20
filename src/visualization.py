import plotly.graph_objects as go
import numpy as np
import torch


def visualize_pcd(points, colors, fig=None, skip=1, size=1., show=True):
    if fig is None:
        fig = go.Figure()
    
    vis_points = points[::skip]
    vis_colors = colors[::skip]
    
    fig.add_trace(go.Scatter3d(
        x=vis_points[:, 0],
        y=vis_points[:, 1],
        z=vis_points[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=vis_colors,
            opacity=1) ))
    
    # Create the layout
    fig.update_layout(
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z')))

    # Show the figure
    if show:
        fig.show()
    return fig

def visualize_cameras(R_c2w, T_c2w, fig=None, show=True, radius = 3, size=1.):
    if fig is None:
        fig = go.Figure()
    
    for r_c2w, t_c2w in zip(R_c2w, T_c2w):
        x = r_c2w[0]
        y = r_c2w[1]
        z = r_c2w[2]
        for v, c, n in zip([x, y, z], ['red', 'green', 'blue'], ['x', 'y', 'z']):
            fig.add_trace(go.Scatter3d(
            x=[t_c2w[0], t_c2w[0] + size*v[0]], 
            y=[t_c2w[1], t_c2w[1] + size*v[1]], 
            z=[t_c2w[2], t_c2w[2] + size*v[2]],
            mode='lines',
            line=dict(color=c, width=max(1, 5)),
            name=n
        ))
    
    # Setting the layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[-radius,radius],),
            yaxis=dict(nticks=4, range=[-radius,radius],),
            zaxis=dict(nticks=4, range=[-radius,radius],),
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Z axis',
            aspectmode='cube'
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10)
    )

    fig.add_trace(go.Scatter3d(
    x=[0],  # x-coordinate for the origin
    y=[0],  # y-coordinate for the origin
    z=[0],  # z-coordinate for the origin
    mode='markers',  # specify the mode as markers to plot points
    marker=dict(
        size=5,  # specify the size of the marker
        color='red',  # specify the color of the marker
    )))
    
    # Update the layout to rotate the plot
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),  # Set Y as the 'up' direction
            eye=dict(x=1, y=1, z=1)  # Adjust the position for better view
        )
    )
    
    # Show plot
    if show:
        fig.show()
    return fig


def visualize_cameras_trajectorie(T_c2w, fig=None, color= 'blue', show=True, radius = 10):
    if fig is None:
        fig = go.Figure()
    
    
    positions = T_c2w.numpy()
    for a, b in zip(positions[1:], positions[:-1]):
        fig.add_trace(go.Scatter3d(
        x=[a[0], b[0]], 
        y=[a[1], b[1]], 
        z=[a[2], b[2]],
        mode='lines',
        line=dict(color=color, width=5, dash='solid'),
    ))
    
    # Setting the layout for better visualization
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=6, range=[-radius,radius],),
            yaxis=dict(nticks=6, range=[-radius,radius],),
            zaxis=dict(nticks=6, range=[-radius,radius],),
            xaxis_title='X axis',
            yaxis_title='Y axis',
            zaxis_title='Z axis',
            aspectmode='cube'
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10)
    )
    
    # Update the layout to rotate the plot
    fig.update_layout(
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),  # Set Y as the 'up' direction
            eye=dict(x=1, y=1, z=1)  # Adjust the position for better view
        )
    )
    
    # Show plot
    if show:
        fig.show()
    return fig