
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



def generate_plot(qTable,L,step,idx:int=0,unwrap:bool=False,numParticles:int = 1):
    fig = plt.figure(figsize=(8, 7))
    plot_title = f'trajectory of particles {idx} up to {numParticles + idx}' + ' unwrapped' if unwrap else ' wrapped'
    ax = fig.add_subplot(111, projection='3d')
    for currParticle in range(idx,numParticles+idx):
        # pull out one trajectory
        q = qTable[::step, currParticle]                
        if unwrap:
            # get all the differences between positions in frames
            d = np.diff(q, axis=0)
            # if a particle moves > L/2 per frame, it's crossed the periodic boundary
            d -= L * np.rint(d / L)
            # adjust accordinly
            q = np.vstack([q[0], q[0] + np.cumsum(d, axis=0)])
        else:
            q = q[1:]
        x, y, z = q.T
        x = x[::step]
        y = y[::step]
        z = z[::step]
        t = np.linspace(0,1, int(len(q)/step))
        ax.scatter(x,y,z,c=t,cmap= "viridis", s = 1,label = f'Trajectory of Particle {idx+currParticle}')
    ax.set(xlabel="x", ylabel="y", zlabel="z", title=plot_title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


