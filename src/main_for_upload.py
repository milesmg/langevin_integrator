# %% [markdown]
# # Here, I build out the Langevin integrator capacities of my system. 

# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
import plotly
import plotly.graph_objects as grapher
import nbformat
# from __future__ import annotations
from typing import Optional, Tuple, List, Literal, Callable
from numpy.typing import NDArray
FloatArray = NDArray[np.floating]

from integrator_scripted import run
from visualizer import generate_video, generate_plot


# %% [markdown]
# # Experiments

# %% [markdown]
# ### Basic First Experiment

# %%
# First experiment: reduced units

rng = np.random.default_rng(42)


N= 100 # number of particles
M = np.ones((N,1),dtype=float) 
dt = 1.e-3
num_steps = 10**4
T = 1.0
gamma = 15.0
kB = 10.0 
L=10**4

# starting positins in a small Gaussian, then center at (0,0,0)
q = 10 * rng.normal(size=(N,3))
q -= q.mean(axis=0, keepdims=True)+ L/2

# starting momenta are boltzmann distributed with stdev = sqrt(mkBT) per component
p_std = np.sqrt(M * kB * T)         
p = 10*p_std * rng.normal(size=(N,3))
# here, we 'center' the momenta so they're not biased in a given direction
p -= p.mean(axis=0, keepdims=True)
p0,q0 = p,q

potential_type = "harmonic"
kSpring=.1


pTable,qTable,UTable,other_Data = run(p0,q0,num_steps=num_steps,
                           N=N,M=M,dt=dt,gamma=gamma,kB=kB,T=T,systemLength=L,
                           potential_type="harmonic",
                           timing = True,printing_steps=True)

print(f"other data = {other_Data}")

# %% [markdown]
# #### Some Simple Timing Experiments: NOT RUNNING
# - Note that I reuse most of the parameters from the first experiment

# %%
timing_experiments = False
if timing_experiments:
    time_table_in_N = []
    Ns = []
    for exp in np.arange(1,6,0.1):
        N_exp = 10**exp
        print(f"N = {N_exp}")
        Ns.append(exp)
        _,_,_,o = run(p0,q0,num_steps=num_steps,
                            N=N_exp,M=M,dt=dt,gamma=gamma,kB=kB,T=T,
                            potential_type="harmonic",
                            timing = True,printing_steps=False)
        time_table_in_N.append(o['time'])

    plt.plot(Ns,time_table_in_N)
    plt.title('Timing vs N')
    plt.xlabel('N')
    plt.ylabel('Time (s)')
    plt.show()

# %% [markdown]
# 

# %% [markdown]
# # Import Visualization Tools

# %%
print(other_Data)

# %%
from visualizer import generate_video, generate_plot

# Visualize doesn't work yet
def visualize(qTable,L,outfile:Optional[str]=None,show:bool= True):
    fig = generate_video(qTable,L,outfile=outfile if outfile else 'outfile.html')
    if show:
        fig.show(renderer= "browser")

def plot(qTable,L,unwrap:bool = False,numParticles:int = 1,idx:Optional[int]=None):
    fig, ax = generate_plot(qTable,L,step=1,idx=0 if idx is None else idx,unwrap=unwrap,numParticles=numParticles)
    plt.show()



# visualize(qTable=qTable,L=10**4,outfile='secondTry.html',show=False)
plot(qTable=qTable,L=10**4,unwrap=True,numParticles=5,idx=10)


