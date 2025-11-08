

import numpy as np
import scipy as sp
import time

from typing import Optional, Tuple, List, Literal, Callable
from numpy.typing import NDArray
FloatArray = NDArray[np.floating]

from numba import njit

#### I realize that I just made this whole parameter-specific function passing useless; 
#### It's just here as a test of numpy vs numba at this point
def get_harmonic_U(k,L):
    def U(q,_):
        difs = q[:, None, :] - q[None, :, :]      # Here, we create a matrix of differences q_i - q_j
        difs -= L * np.round(difs / L)             
        sumSquares = np.sum(difs**2,axis=2) # sum the x,y, and z components to get an N x N matrix

        return(0.5*k*np.sum(np.triu(sumSquares,k=1))) # we only care about the unique pairs
    return(U) 

def get_harmonic_Grad(k,L):
    def Grad(q):
        difs = q[:, None, :] - q[None, :, :]      # See function above for explanatino
        difs -= L * np.round(difs / L)  
        return(k* np.sum(difs,axis = 1)) # sum over j, noting that the dif for j=i is 0
    return(Grad) 


@njit(cache=True, fastmath=True)
def harmonic_U_numba(q,params):
    # pass in params of the form k,L
    k = params[0]
    L = params[1]
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
            djx = q[j,0]-qix
            djy = q[j,1]-qiy
            djz = q[j,2]-qiz            
            if djx>  half:  djx -= L
            elif djx <= -half: djx += L
            if djy >  half:  djy -= L
            elif djy <= -half: djy += L
            if djz >  half:  djz -= L
            elif djz <= -half: djz += L
            sumSquares += djx**2 + djy**2 + djz**2
    return (0.5 *k * sumSquares)

@njit(cache=True, fastmath=True)
def harmonic_Grad_numba(q,k,L):
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
@njit(cache=True, fastmath=True)
def apply_A_numba(p,q,dt,invM):
    N=q.shape[0]
    for idx in range(N):
        coeff = 1*invM[idx] * dt
        q[idx,0] += p[idx,0]* coeff
        q[idx,1] += p[idx,1]* coeff
        q[idx,2] += p[idx,2]* coeff
    # these are in place operations; don't need to return anything

def apply_A(p,q,dt,M):
    q += p/M * dt 
    #  these operations are in-place; I don't need to return anything

# Two Choices for B: 

@njit(cache=True, fastmath=True)
def apply_B_numba(p,q,dt,gradU):
    # the B operator applies the newtonian force kick 
    F_neg = gradU(q)
    N=q.shape[0]
    for idx in range(N):
        p[idx,0] += -F_neg[idx,0] * dt
        p[idx,1] += -F_neg[idx,1] * dt
        p[idx,2] += -F_neg[idx,2] * dt

@njit(cache=True, fastmath=True)
def apply_harmonic_B_numba(p,q,params):
    k = params[0]
    L = params[1]
    dt = params[2]
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

def apply_B(p,q,params):
    dt,getGrad = params[0],params[1]
    # the B operator applies the newtonian force kick 
    F = -getGrad(q)
    p += F*dt

# for now, I'd rather not deal with how numba handles randomness; anyway this step is O(N)
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
        add_noise:bool=True,numbaify = False):

    other_data = {}

    if timing:
        other_data['B_time'] = 0
        other_data['A_time'] = 0
        other_data['O_time'] = 0
        other_data['save_time'] = 0
        other_data['U_time'] = 0



    A_operation = apply_A_numba if numbaify else apply_A
    O_operation = apply_O
    if not numbaify: B_operation = apply_B
    if potential_type == "harmonic":
        if not numbaify:
            getU = get_harmonic_U(kSpring,systemLength)
            U_params = None
            getGrad = get_harmonic_Grad(kSpring,systemLength)
            B_params = [dt/2,getGrad]
        else:
            getU=harmonic_U_numba
            U_params = np.array([kSpring,systemLength])
            B_operation = apply_harmonic_B_numba
            B_params = np.array([kSpring,systemLength,dt/2])
  

    # this was tripping me up
    # i need to enforce the memory structure I want
    q = np.ascontiguousarray(q0, dtype=np.float64)
    p = np.ascontiguousarray(p0, dtype=np.float64)

    M = np.asarray(M, dtype=np.float64)
    if M.ndim == 1:
        M = M.reshape(N, 1)
    M = np.ascontiguousarray(M)
    invM1d = 1.0 / M[:, 0]


    q_table,p_table,U_table = [q.copy()],[p.copy()],[getU(q,U_params)]

    def save_data(p,q,saveU = False,timing = timing):
        q_table.append(q.copy())
        p_table.append(p.copy())
        if saveU:
            if timing: U_time = time.perf_counter()
            U_table.append(getU(q,U_params))
            if timing: other_data['U_time'] += time.perf_counter()-U_time 


    def BAOAB(p,q,timing = False,other_data =other_data,add_noise=add_noise):
    # perform BAOAB
    # 1st B
        if timing: clock= time.perf_counter()
        B_operation(p,q,B_params)
        if timing:other_data['B_time'] += time.perf_counter()-clock
    # 1st A
        if timing: clock = time.perf_counter()
        A_operation(p,q,dt/2,invM1d if numbaify else M)
        # apply periodic boundary conditions
        q[:]= np.mod(q,systemLength)
        if timing:other_data['A_time'] +=time.perf_counter()-clock
    # O
        if add_noise:
            if timing: clock = time.perf_counter()
            O_operation(p,q,dt,gamma,M,kB,T,rng)
            if timing:other_data['O_time'] += time.perf_counter()-clock
    # 2nd A
        if timing: clock = time.perf_counter()
        A_operation(p,q,dt/2,invM1d if numbaify else M)
        # apply periodic boundary conditions
        q[:] = np.mod(q,systemLength)
        if timing:other_data['A_time'] +=time.perf_counter()-clock
    ## 2nd B
        if timing: clock= time.perf_counter()
        B_operation(p,q,B_params)
        if timing:other_data['B_time'] += time.perf_counter()-clock
        
    

    #### MAIN LOOP ####
    if timing: start = time.perf_counter()
    for step in range(num_steps):
        if printing_steps and step>0 and step % int(num_steps//10) == 0:
                print(f"At step {step} out of {num_steps}")
        BAOAB(p,q,timing = timing)
        if timing: save_clock = time.perf_counter()
        save_data(p,q,saveU=saveU)
        if timing: other_data['save_time'] += time.perf_counter() - save_clock
        
    if timing: other_data['time'] = time.perf_counter()-start
    other_data['systemLength'] = systemLength
    other_data['T'] = T
    other_data['kB'] = kB
    other_data['gamma'] = gamma
    other_data['dt'] = dt
    other_data['M'] = M
    other_data['numbaified'] = numbaify
    other_data['potential_type'] = potential_type

    
    return (np.array(p_table),np.array(q_table),np.array(U_table),other_data)
