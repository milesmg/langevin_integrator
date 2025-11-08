


# Langevin Integrator

## Background / Motivation

I've always been interested in computational methods for solving chemistry problems, and this is just about the most basic. Since Einstein, the question of 'how do molecules move' has had various answers, but the most important ones you run into today all revolve around computational methods of discretizing the equations of motion at the molecular scale. Such discretizations run into a few major challenges
- Ergodicity/Bias: You want your system to sample all the key microstates it should in the correct proportion, and you want this to happen quickly enough that you can get there in one simulation.
- Physical Accuracy: You need your system to be well represented, particularly when it comes to respecting conservation laws. 
- Efficiency: You need to run your code on a real computer.

Here, I will work with the BAOAB implementation of Langevin dynamics (one formualation of stochastic MD). This project is based on the landmark paper 'Robust and efficient configurational molecular sampling via Langevin dynamics' by Benedict Leimkuhler and Charles Matthews. 

One of the key dilemmas the authors describe is choosing a step size for your numerical integrator. Too large a step size will make the integrator inaccurate, but too small a step size slows down your sampling. Here, the authors are very interested in multiscale modeling--that is, integrators that can handle different timesteps with relatively strong error bounds. 

This paper came a few years after the original OpenMM/GPU-MD paper(s), so it's cool to see them already working with NAMD. 

## Physics and Computation

Langevin dynamics is a clever implementation of standard mechanics in a heat bath. For a given particle, we have $$\begin{aligned}x' = v\end{aligned} \\ mv' = -\nabla U - \gamma v+ \xi(t)\sqrt{2\gamma k_B T m},$$ where $\gamma$ is a friction coefficient and $\xi(t)$ is random noise. The beauty of this equation is that it 
- keeps T fixed without manual adjustment
- in the long run produces configurations drawn from a Boltzmann distribution, as desired
- can be tuned with $\gamma$. 

In the paper, the authors focus on splitting methods for numerical implementations of Langevin dynamics. Splitting methods get around the problem of the analytical intractability of these equations. It is impossible to solve these integrals in an elementary manner, and it is numerically quite inefficient to try to one-shot them. Rather, as we are already breaking up our simulations into timesteps, it makes sense to apply individual components of our Langevin dynamics separately, allowing for simpler computation. Observe that we can break the process above into the system of equations $$\begin{aligned} dq =& M^{-1}pdt \\ dp =& -\nabla U(q) dt + (-\gamma p + \xi(t) \sqrt{2\gamma k_B Tm})dt \end{aligned}$$ where $p,q \in \mathbb{R}^{3N}$ are the state vectors for our $N$ particles. We can split this into three pieces: $$\begin{aligned}A =& [M^{-1}p \ \ 0]^Tdt \\ B=&  [0 \ \ -\nabla U(q) ]^Tdt \\ O =&  [0 \ \ (-\gamma p + \xi(t) \sqrt{2\gamma k_B Tm}) ]^Tdt\end{aligned}$$ which we apply to our system $[dq \ \  dp]^T$. We then apply these split operations in whatever order we choose, so long as the timesteps we afford each operation average out to equivalence. The order that the authors of this paper settle on is **BAOAB**. 

## To Do:

#### Notes:
- Two major issues: 
    - I think there's a sign error in my forces that is making things attractive rather than repulsive.
    - I need to decide what to numba/njit-ify

#### Functionality:
- build a potential matrix
- build some kind of box 
- add periodic boundary conditions


#### Aesthetics
- store data in SQL table
- build some kind of visualization
- build into multiple python/cython/c/sql etc. files
    - main: sets parameters, runs simulation
    - integrator: the engine; runs all the computations
    - visualizer: builds the visualizations
    - data processer: builds database of experiments

#### Optimization

- improve the O step with better numerics
- make this compiled code (cython?) or re-write in c++