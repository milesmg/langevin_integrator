# %% [markdown]
# # Here, I build out the Langevin integrator capacities of my system. 

# %%
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as grapher
import nbformat
from typing import Optional, Tuple, List, Literal, Callable
from numpy.typing import NDArray

from typing_extensions import final
from mpl_toolkits.mplot3d import Axes3D
FloatArray = NDArray[np.floating]


# %% [markdown]
# # Visualization

# %%
# This is mostly ChatGPT code; I'm not a plotly expert. 

def generate_video(qTable:Optional[FloatArray] = None, L:Optional[int]=None, trail:int=1, stride:int=1, title:str ="Visualizing", 
            point_size:int = 3, box=True, outfile:str ='outfile.md'):
    
    # going to plot everything
    # using a trace so things are more visualizable
    # stride will choose which frames
    # obviously title is title, outfile is outfile
    # box is boundaries
    # point_size is size of particles
    

    q = qTable[::stride]
    T, N, _ = q.shape

    # Build traces for the current frame + trail
    def frame_data(t):
        traces = []
        # here we build our trace
        traces.append(grapher.Scatter3d(
            x=q[t,:,0], y=q[t,:,1], z=q[t,:,2],
            mode="markers",
            marker=dict(size=point_size),
            name="t",
            showlegend=False
        ))
        # trail (fading)
        kmax = min(trail, t)
        for k in range(1, kmax+1):
            # add our trace with the right magnitude
            alpha = 1.0 - k/(kmax+1)
            traces.append(grapher.Scatter3d(
                x=q[t-k,:,0], y=q[t-k,:,1], z=q[t-k,:,2],
                mode="markers",
                # gets more translucent as we get further from the current point
                marker=dict(size=max(1, point_size-1), opacity=alpha),
                hoverinfo="skip",
                showlegend=False
            ))
        return traces


    # First frame
    data0 = frame_data(0)

    # Frames
    frames = [grapher.Frame(data=frame_data(t), name=str(t)) for t in range(T)]

    # Scene/layout
    axes = dict(range=[0, L], showgrid=False, zeroline=False, title='')
    layout = grapher.Layout(
        title=title,
        scene=dict(xaxis=axes, yaxis=axes, zaxis=axes, aspectmode="cube"),
        width=800, height=700,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "y": 0,
                "x": 0,
                "xanchor": "left",
                "buttons": [
                    {"label": "Play", "method": "animate",
                     "args": [None, {"frame": {"duration": 30, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]},
                    {"label": "Pause", "method": "animate",
                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}]}
                ],
            }
        ],
        sliders=[{
            "steps": [{"args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                           "mode": "immediate"}],
                       "label": str(k), "method": "animate"} for k in range(T)],
            "currentvalue": {"prefix": "frame: "}
        }]
    )

    fig = grapher.Figure(data=data0, frames=frames, layout=layout)

    # Optional: wireframe box
    if box:
        corners = np.array([[0,0,0],[L,0,0],[L,L,0],[0,L,0],[0,0,0],
                            [0,0,L],[L,0,L],[L,L,L],[0,L,L],[0,0,L]])
        edges = [(0,1),(1,2),(2,3),(3,0),(5,6),(6,7),(7,8),(8,5),(0,5),(1,6),(2,7),(3,8)]
        for a,b in edges:
            fig.add_trace(grapher.Scatter3d(
                x=[corners[a,0], corners[b,0]],
                y=[corners[a,1], corners[b,1]],
                z=[corners[a,2], corners[b,2]],
                mode="lines",
                line=dict(width=2),
                hoverinfo="skip",
                showlegend=False
            ))

    if outfile:
        fig.write_html(outfile, include_plotlyjs="cdn")
    return fig



# %%
def generate_plot(qTable,L,step,idx:int=0,unwrap:bool=False):
    if unwrap:
        """Reconstructs a continuous trajectory from wrapped coordinates."""
        d = np.diff(qTable[:,idx], axis=0)
        d -= L * np.round(d / L) # Minimum image convention on the *difference*
        [x,y,z]=np.vstack([qTable[:,idx,0], qTable[:,idx,0] + np.cumsum(d, axis=0)])
    else:
        [x,y,z] = qTable[:,idx]


    fig = plt.figure(figsize=(8, 7))
    plot_title = f'trajectory of particle {idx}' + 'unwrapped' if unwrap else 'wrapped'
    ax = fig.add_subplot(111, projection='3d')
    div = 5000
    x = x[::step]
    y = y[::step]
    z = z[::step]
    ax.plot(x,y,z)
    t = np.linspace(0,1, int(len(qTable)/div))
    scatter = ax.scatter(x,y,z,c=t,cmap= "viridis", s = 1)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=plot_title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


