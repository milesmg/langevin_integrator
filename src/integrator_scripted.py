# %% [markdown]

import numpy as np
import scipy as sp
import time

from typing import Optional, Tuple, List, Literal, Callable
from numpy.typing import NDArray
FloatArray = NDArray[np.floating]


def get_harmonic_U(k,L):
    def U(q):
        difs = q[:, None, :] - q[None, :, :]      # Here, we create a matrix of differences q_i - q_j
        difs -= L * np.round(difs / L)             
        sumSquares = np.sum(difs**2,axis=2) # sum the x,y, and z components to get an N x N matrix

        return(0.5*k*np.sum(np.triu(sumSquares,k=1))) # we only care about the unique pairs
    return(U) 

def get_harmonic_Nabla(k,L):
    def Nabla(q):
        difs = q[:, None, :] - q[None, :, :]      # See function above for explanatino
        difs -= L * np.round(difs / L)  
        return(k* np.sum(difs,axis = 1)) # sum over j, noting that the dif for j=i is 0
    return(Nabla) 


def apply_A(p,q,dt,M):
    # the A operator doesn't touch p, but it adjusts q (position) via Newton
    q += p/M * dt # careful! numpy will broadcast M.shape = (N,1) but not (N,)... need to keep an eye out
    #  these operations are in-place; I don't need to return anything

def apply_B(p,q,dt,getNablaU):
    # the B operator applies the newtonian force kick 
    F = -getNablaU(q)
    p += F*dt

def apply_O(p,q,dt,gamma,M,kB,T,rng):

    a = np.exp(-gamma*dt)
    one_minus_a2 = np.expm1(-2*gamma*dt) * (-1)   # equals 1 - a**2 with better accuracy
    one_minus_a2 = np.clip(one_minus_a2, 0.0, 1.0) # clip to avoid more floating point errors
    stdev = np.sqrt(M*kB*T*(one_minus_a2))
    xi = rng.normal(size = p.shape)
    p *= a
    p += stdev * xi 



def run(p0:Optional[FloatArray] = None,q0:Optional[FloatArray] = None,N: int = 10,
        M:Optional[FloatArray]=None,
        dt:float = 1.e-3,num_steps:int = 10*3,T:float = 1.0,gamma:float = 1.0,kB:float = 1.0,systemLength:float = 10**4,
        rng = np.random.default_rng(42),
        potential_type = "harmonic",kSpring:float = 1.0,
        timing = False, printing_steps = False,
        saveU= False,
        add_noise:bool=True):

    other_data = {}

    if timing:
        other_data['B_time'] = 0
        other_data['A_time'] = 0
        other_data['O_time'] = 0
        other_data['save_time'] = 0
        other_data['U_time'] = 0



    if potential_type == "harmonic":
        getNabla = get_harmonic_Nabla(kSpring,systemLength)
        getU = get_harmonic_U(kSpring,systemLength)

    p=p0
    q=q0
    if M.ndim == 1:
        M = M.reshape(N, 1)


    q_table,p_table,U_table = [q.copy()],[p.copy()],[getU(q)]

    def save_data(p,q,saveU = False,timing = timing):
        q_table.append(q.copy())
        p_table.append(p.copy())
        if saveU:
            if timing: U_time = time.perf_counter()
            U_table.append(getU(q))
            if timing: other_data['U_time'] += time.perf_counter()-U_time 


    def BAOAB(p,q,timing = False,other_data =other_data,add_noise=add_noise):
    # perform BAOAB
    # 1st B
        if timing: clock= time.perf_counter()
        apply_B(p,q,dt/2,getNabla)
        if timing:other_data['B_time'] += time.perf_counter()-clock
    # 1st A
        if timing: clock = time.perf_counter()
        apply_A(p,q,dt/2,M)
        # apply periodic boundary conditions
        q = np.mod(q,systemLength)
        if timing:other_data['A_time'] +=time.perf_counter()-clock
    # O
        if add_noise:
            if timing: clock = time.perf_counter()
            apply_O(p,q,dt,gamma,M,kB,T,rng)
            if timing:other_data['O_time'] += time.perf_counter()-clock
    # 2nd A
        if timing: clock = time.perf_counter()
        apply_A(p,q,dt/2,M)
        # apply periodic boundary conditions
        q = np.mod(q,systemLength)
        if timing:other_data['A_time'] +=time.perf_counter()-clock
    ## 2nd B
        if timing: clock= time.perf_counter()
        apply_B(p,q,dt/2,getNabla)
        if timing:other_data['B_time'] += time.perf_counter()-clock
        
        return(p,q)
    

    #### MAIN LOOP ####
    if timing: start = time.perf_counter()
    for step in range(num_steps):
        if printing_steps and step>0 and step % int(num_steps//10) == 0:
                print(f"At step {step} out of {num_steps}")
        p,q = BAOAB(p,q,timing = timing)
        if timing: save_clock = time.perf_counter()
        save_data(p,q,saveU=saveU)
        if timing: other_data['save_time'] += time.perf_counter() - save_clock
        
    if timing: other_data['time'] = time.perf_counter()-start
    other_data['systemLength'] = systemLength
    other_data['T'] = T
    other_data['kB'] = kB
    other_data['gamma'] = gamma
    other_data['dt'] = dt

    
    return (np.array(p_table),np.array(q_table),np.array(U_table),other_data)


