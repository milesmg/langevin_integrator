# %% [markdown]

import numpy as np
import scipy as sp
import time

from typing import Optional, Tuple, List, Literal, Callable
from numpy.typing import NDArray
FloatArray = NDArray[np.floating]


def harmonic_U(q,k,L):
    N=q.shape[0]
    half = 0.5 * L
    sumSquares = 0.0
    # the thing about writing good numba code is that it's actually just
    # the worst code you've ever written--code that a sixth grade would write--with 
    # @numba added before the function definition. 
    for i in range(N-1):
        qix = q[i,0]
        qiy=q[i,1]
        qiz = q[i,2]
        for j in range(i+1, N):
            djx = q[j][0]-qix
            djy = q[j][1]-qiy
            djz = q[j][2]-qiz            
            if djx>  half:  djx -= L
            elif djx <= -half: djx += L
            if djy >  half:  djy -= L
            elif djy <= -half: djy += L
            if djz >  half:  djz -= L
            elif djz <= -half: djz += L
            sumSquares += djx**2 + djy**2 + djz**2
    return (0.5 *k * sumSquares)


def harmonic_Grad(q,k,L):
    N = q.shape[0]
    half = 0.5*L
    grad = np.zeros(q.shape,dtype=np.float64)
    for i in range(N):
        qix = q[i,0]
        qiy=q[i,1]
        qiz = q[i,2]
        # I think I could do something clever here
        # so that I don't have to loop N^2 but N^2/2; if I've
        # already calculated q[j]-q[i], i shouldn't need q[i]-q[j]
        for j in range(N):
            djx = qix-q[j,0]
            djy = qiy-q[j,1]
            djz = qiz-q[j,2]    
            if djx>  half:  djx -= L
            elif djx <= -half: djx += L
            if djy >  half:  djy -= L
            elif djy <= -half: djy += L
            if djz >  half:  djz -= L
            elif djz <= -half: djz += L
            # i reversed the signs :(
            grad[i,0] += k*djx
            grad[i,1] += k*djy
            grad[i,2] += k*djz
    return(grad)


# This is a nice test case for numba vs numpy on O(N)
def apply_A_numba(p,q,dt,M):
    N=q.shape[0]
    for idx in range(N):
        coeff = 1/M[idx] * dt
        q[idx,0] += p[idx,0]* coeff
        q[idx,1] += p[idx,1]* coeff
        q[idx,2] += p[idx,2]* coeff
    # these are in place operations; don't need to return anything

def apply_A(p,q,dt,M):
    q += p/M * dt 
    #  these operations are in-place; I don't need to return anything

# Two Choices for B: 

def apply_B_numba(p,q,dt,gradU):
    # the B operator applies the newtonian force kick 
    F_neg = gradU(q)
    N=q.shape[0]
    for idx in range(N):
        p[idx,0] += -F_neg[idx,0] * dt
        p[idx,1] += -F_neg[idx,1] * dt
        p[idx,2] += -F_neg[idx,2] * dt

def apply_harmonic_B_numba(p,q,k,L,dt):
    N = q.shape[0]
    half = 0.5*L
    # grad = np.zeros(q.shape,dtype=np.float64)
    for i in range(N):
        qix = q[i,0]
        qiy=q[i,1]
        qiz = q[i,2]
        # I think I could do something clever here
        # so that I don't have to loop N^2 but N^2/2; if I've
        # already calculated q[j]-q[i], i shouldn't need q[i]-q[j]
        gradix = 0.0; gradiy = 0.0; gradiz = 0.0
        for j in range(N):
            djx = qix-q[j,0]
            djy = qiy-q[j,1]
            djz = qiz-q[j,2]           
            if djx>  half:  djx -= L
            elif djx <= -half: djx += L
            if djy >  half:  djy -= L
            elif djy <= -half: djy += L
            if djz >  half:  djz -= L
            elif djz <= -half: djz += L
            # i reversed the signs :(
            gradix += k*djx
            gradiy += k*djy
            gradiz += k*djz
        p[i,0] += -gradix *dt
        p[i,1] += -gradiy *dt
        p[i,2] += -gradiz *dt


# for now, I'd rather not deal with how numba handles randomness; anyway this step is O(N)
def apply_O(p,q,dt,gamma,M,kB,T,rng):
    a = np.exp(-gamma*dt)
    one_minus_a2 = np.expm1(-2*gamma*dt) * (-1)   # equals 1 - a**2 with better accuracy
    one_minus_a2 = np.clip(one_minus_a2, 0.0, 1.0) # clip to avoid more floating point errors
    stdev = np.sqrt(M*kB*T*(one_minus_a2))
    xi = rng.normal(size = p.shape)
    p = a*p + stdev * xi



def run_numba(p0:Optional[FloatArray] = None,q0:Optional[FloatArray] = None,N: int = 10,
        M:Optional[FloatArray]=None,
        dt:float = 1.e-3,num_steps:int = 10*3,T:float = 1.0,gamma:float = 1.0,kB:float = 1.0,systemLength:float = 10**4,
        rng = np.random.default_rng(42),
        potential_type = "harmonic",kSpring:float = 1.0,
        timing = False, printing_steps = False):
    
    other_data = {}


    if potential_type == "harmonic":
        getNabla = get_harmonic_Nabla(kSpring,systemLength)
        getU = get_harmonic_U(kSpring,systemLength)

    p=p0
    q=q0


    q_table,p_table,U_table = [q.copy()],[p.copy()],[getU(q)]

    def save_data(p,q):
        q_table.append(q.copy())
        p_table.append(p.copy())
        U_table.append(getU(q))


    def BAOAB(p,q):
        # perform BAOAB
        p,q = apply_B(p,q,dt/2,getNabla)

        p,q = apply_A(p,q,dt/2,M)
        # apply periodic boundary conditions

        q = np.mod(q,systemLength)
        p,q = apply_O(p,q,dt,gamma,M,kB,T,rng)
        p,q = apply_A(p,q,dt/2,M)

        # apply periodic boundary conditions
        q = np.mod(q,systemLength)

        p,q = apply_B(p,q,dt/2,getNabla)
        return(p,q)


    if timing: 
        start = time.perf_counter()

    #### MAIN LOOP ####
    for step in range(num_steps):

        if printing_steps and step>0 and step % int(num_steps//10) == 0:
                print(f"At step {step} out of {num_steps}")

        p,q = BAOAB(p,q)
        save_data(p,q)
        
    
    if timing: 
        end = time.perf_counter()
        other_data['time'] = end-start
    other_data['systemLength'] = systemLength
    other_data['T'] = T
    other_data['kB'] = kB
    other_data['gamma'] = gamma
    other_data['dt'] = dt

    
    return (np.array(p_table),np.array(q_table),np.array(U_table),other_data)


