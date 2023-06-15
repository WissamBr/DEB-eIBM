### Imports ###
import numpy as np
from test_individual__ import Individual
import math
from numba import jit
from scipy.integrate import solve_ivp
import pandas as pd
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.pyplot as plt

#%%
def plot_occupant_clip(grids, total_time, fps):
    '''
    Plots a clip of the spatial distribution of individuals in grids for total_time frames with a frame rate of fps.
    '''
    fig, ax = plt.subplots()
    ca_plot = ax.imshow(grids['occupant'][0,:,:], cmap=colors.ListedColormap(['k','y','r','b']), vmin=0, vmax=3) #cmap='Reds', vmin=0, vmax=5)#cmap=colors.ListedColormap(['k','y','r','b']), vmin=0, vmax=3)
    
    def animation_func(frame_num):
        ca_plot.set_data(grids['occupant'][frame_num,:,:])
        return ca_plot
    
    
    anim = FuncAnimation(fig, animation_func, frames=total_time, interval=fps)
    plt.show()
    return anim
        
#%%
def plot_food_clip(grids, total_time, fps):
    '''
    Plots a clip of the spatial distribution of food in grids for total_time frames with a frame rate of fps.
    '''
    fig,ax = plt.subplots()
    ca_plot = ax.imshow(grids['food'][0,:,:], vmin=0, vmax=np.max(grids["food"]))#cmap=colors.ListedColormap(['k','y','r','b']), vmin=0, vmax=3)
    
    def animation_func(i):
        ca_plot.set_data(grids['food'][i,:,:])
        return ca_plot
    
    anim = FuncAnimation(fig, animation_func, frames=total_time, interval=fps)
    plt.show()
    return anim

#%%
@jit(nopython=True)
def beta0(x0, x1):
    '''
    Calculates the integral of the function beta(x) from x0 to x1.
    '''
    x03 = x0**(1/ 3)
    x13 = x1**(1/ 3)
    a3 = math.sqrt(3)

    f1 = - 3 * x13 + a3 * (np.arctan((1 + 2 * x13)/ a3) - np.arctan(1/a3)) - np.log(1 - x13) + np.log(1 + x13 + x13**2)/ 2
    f0 = - 3 * x03 + a3 * (np.arctan((1 + 2 * x03)/ a3) - np.arctan(1/a3)) - np.log(1 - x03) + np.log(1 + x03 + x03**2)/ 2
    f = f1 - f0
    return f

### DEB-related functions ###
@jit(nopython=True)
def init_reserve_DEB(DEB_p, f_resp):
    '''
    Calculates the initial reserve of an individual based on its DEB parameters and the functional response experienced by its parent or the reserve density of the parent.
    We use the Newton-Raphson method to find the scaled length at birth, similar to the get_lb function in DEBtool.
    '''
    
    # Unpack the parameters of the individual
    p_Am = DEB_p[2]*DEB_p[1]
    E_Hb = DEB_p[12]

    v_dot = DEB_p[5] # (-)
    kap = DEB_p[6] # (-)
    p_M = DEB_p[8] # (J cm-3 d-1)
    k_J = DEB_p[10] # (d-1)
    E_G = DEB_p[11] # (J cm-3)
    
    # check for viability
    if p_Am**3*E_G*(1-kap)*kap**2/p_M**3 < E_Hb:
        return 0
    
    # Determine the functional response and the offspring projected reserve density
    f = f_resp
    eb = f

    # Determine the necessary compound parameters
    g = E_G * v_dot / (kap * p_Am) # investment ratio

    k_M = p_M/E_G
    k = k_J/k_M # maintenance ratio

    v_Hb = E_Hb * g**2 * k_M**3 / ((1-kap) * v_dot**2 * p_Am)


    # Initialize the algorithm for finding scaled length at birth
    ns = 1000 + round(1000 * max(0, k - 1)) 
    xb = g / (g + eb)
    xb3 = xb ** (1/3)
    x = np.linspace(1e-6, xb, ns)
    dx = xb / ns
    x3 = x**(1/3)

    b = beta0(x, xb)/(3 * g)
    t0 = xb * g * v_Hb

    lb = v_Hb ** (1/3) #Initialization of length at birth
    # Starting procedure for Newton-Raphson method
    i = 0
    norm = 1

    # Maximum number of iterations
    ni = 1000

    # Algorithm for scaled length at birth (~get_lb from DEBtool)
    while (i < ni)  & (norm > 1e-8):
        l = x3 / (xb3/ (lb - b))
        s = (k - x) / (1 - x) * l/ g / x
        v = math.e**( - dx * np.cumsum(s))
        if np.any(v == 0):
            break
        vb = v[ns-1]
        r = (g + l)
        rv = r / v
        t = t0/ lb**3/ vb - dx * np.sum(rv)
        dl = xb3/ lb**2 * l**2 / x3
        dlnl = dl / l
        dv = v * math.e**(-dx*np.cumsum(s*dlnl))
        dlnv = dv / v
        dlnvb = dlnv[ns-1]
        dr = dl
        dlnr = dr / r
        dt = - t0/ lb**3/ vb * (3/ lb + dlnvb) - dx * np.sum((dlnr - dlnv) * rv)
        lb = lb - t/ dt # Newton Raphson step
        norm = t**2
        i = i + 1
    
    if np.any(v==0):
        return 0
    
    # Scaled initial reserve
    uE0 = (3 * g/ (3 * g * xb**(1/ 3)/ lb - beta0(0.0, xb)))**3
    E0 = uE0 * v_dot**2 / (g**2 * k_M**3) * p_Am

    # negative initial reserve turns zero
    if E0 < 0:
        E0 = 0

    return E0

#%%

@jit(nopython=True)
def DEB_normal_em(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables of an embryo and food (y) based on its DEB parameters and temperature.

    Parameters
    ----------
    t : float
        Time (d).
    y : array
        State variables of the embryo and food (L, E, E_H, E_R, q, h, X).
    DEB_p : array
        DEB parameters of the embryo and food (p_Am, E_m, L_m, W_m, kap_X, v, kap, p_M, p_T, k_J, E_G, E_Hb, E_Hj, E_Hp, T_A, T_ref, h_a, s_G, T_L, T_H, T_AL, T_AH).
    temperature : float
        Temperature (K).

    Returns
    -------
    dydt : array
        Change in state variables of the embryo and food (dLdt, dEdt, dE_Hdt, dE_Rdt, dqdt, dhdt, dXdt).
    '''
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    kap_X = DEB_p[4]                            # digestion efficiency (J assimilated/ J ingested)
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    k_J = DEB_p[10]                             # maturity maintenance rate coefficient (d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Assimilation flux: if the organism is an embryo, it can't feed by itself
    power_A = 0

    # Feeding flux
    power_X = power_A/kap_X

    # Mobilisation flux
    power_C = ( (E/L**3) * (v*E_G*L**2 + p_M*L**3 + p_T*L**2)/(E_G + kap*(E/L**3)) )*arrhenius_corr

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = kap*power_C - power_S

    # Maturity maintenance
    power_J = (k_J*E_H)*arrhenius_corr

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = 0
    power_H = (1-kap) * power_C - power_J
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (power_A - power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = (-power_X)

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEBcat_normal_juv(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables of a juvenile, without ingestion and assimilation processes, based on its DEB parameters and temperature.

    Parameters
    ----------
    t : float
        Time (d).
    y : array
        Array containing the state variables of the juvenile.
    DEB_p : array
        Array containing the DEB parameters of the juvenile.
    temperature : float
        Temperature (K).

    Returns
    -------
    dydt : array
        Array containing the change in DEB state variables of the juvenile.
    '''

    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    k_J = DEB_p[10]                             # maturity maintenance rate coefficient (d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Mobilisation flux
    power_C = ( (E/L**3) * (v*E_G*L**2 + p_M*L**3 + p_T*L**2)/(E_G + kap*(E/L**3)) )*arrhenius_corr

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = kap*power_C - power_S

    # Maturity maintenance
    power_J = (k_J*E_H)*arrhenius_corr

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = 0
    power_H = (1-kap) * power_C - power_J
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (- power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = 0

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEB_normal_juv(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables for a juvenile organism and food availability based on the DEB parameters and temperature.

    Parameters
    ----------
    t : float
        Time.
    y : array
        Array containing the state variables.
    DEB_p : array
        Array containing the DEB parameters.
    temperature : float
        Temperature.
    
    Returns
    -------
    dydt : array
        Array containing the change in state variables.
    '''
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    F_m = DEB_p[3]                              # surface-area specific searching rate (cells cm-2 d-1)
    kap_X = DEB_p[4]                            # digestion efficiency (J assimilated/ J ingested)
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    k_J = DEB_p[10]                             # maturity maintenance rate coefficient (d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    mu_X = DEB_p[18]
    mu_E = DEB_p[19]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Feeding conversion
    y_E_X = kap_X * mu_X/mu_E
    y_X_E = 1/y_E_X
    
    K = (p_Am*y_X_E)/F_m # Half saturation constant in terms of energy food
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    
    # Definition functional response & making sure it's zero when food runs out
    if X <= 0:
        X = 0
        
    f = (X/(K+X))

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Assimilation flux: if the organism is an embryo, it can't feed by itself
    power_A = (p_Am*f*L**2)*arrhenius_corr

    # Feeding flux
    power_X = power_A*y_X_E

    # Mobilisation flux
    power_C = ( (E/L**3) * (v*E_G*L**2 + p_M*L**3 + p_T*L**2)/(E_G + kap*(E/L**3)) )*arrhenius_corr

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = kap*power_C - power_S

    # Maturity maintenance
    power_J = (k_J*E_H)*arrhenius_corr

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = 0
    power_H = (1-kap) * power_C - power_J
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (power_A - power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = -power_X

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEBcat_normal_adu(t, y, DEB_p, temperature):
    '''
    Calculate the change in DEB state variables for an adult organism, without ingestion and assimilation processes, based on the DEB parameters and the current state variables
    
    Parameters
    ----------
    t : float
        Current time
    y : array
        Current state variables
    DEB_p : array
        DEB parameters
    temperature : float
        Current temperature

    Returns
    -------
    dydt : array
    '''
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    k_J = DEB_p[10]                             # maturity maintenance rate coefficient (d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Mobilisation flux
    power_C = ( (E/L**3) * (v*E_G*L**2 + p_M*L**3 + p_T*L**2)/(E_G + kap*(E/L**3)) )*arrhenius_corr

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = kap*power_C - power_S

    # Maturity maintenance
    power_J = (k_J*E_H)*arrhenius_corr

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = (1-kap) * power_C - power_J
    power_H = 0
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (- power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = 0

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEB_normal_adu(t, y, DEB_p, temperature):
    '''
    Calculate the change in DEB state variables for an adult organism and the food availability based on the DEB parameters and the current state variables.

    Parameters
    ----------
    t : float
        Current time.
    y : array
        Current state variables.
    DEB_p : array
        DEB parameters.
    temperature : float
        Current temperature.

    Returns
    -------
    dydt : array
        Change in state variables.
    '''
    p_Am = DEB_p[2]*DEB_p[1]
    F_m = DEB_p[3]                              # surface-area specific searching rate (cells cm-2 d-1)
    kap_X = DEB_p[4]                            # digestion efficiency (J assimilated/ J ingested)
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    k_J = DEB_p[10]                             # maturity maintenance rate coefficient (d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    mu_X = DEB_p[18]
    mu_E = DEB_p[19]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Feeding conversion
    y_E_X = kap_X * mu_X/mu_E
    y_X_E = 1/y_E_X
    
    K = (p_Am*y_X_E)/F_m # Half saturation constant in terms of energy food
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    
    # Definition functional response & making sure it's zero when food runs out
    if X <= 0:
        X = 0
        
    f = (X/(K+X))

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Assimilation flux: if the organism is an embryo, it can't feed by itself
    power_A = (p_Am*f*L**2)*arrhenius_corr

    # Feeding flux
    power_X = power_A*y_X_E

    # Mobilisation flux
    power_C = ( (E/L**3) * (v*E_G*L**2 + p_M*L**3 + p_T*L**2)/(E_G + kap*(E/L**3)) )*arrhenius_corr

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = kap*power_C - power_S

    # Maturity maintenance
    power_J = (k_J*E_H)*arrhenius_corr

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = (1-kap) * power_C - power_J
    power_H = 0
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (power_A - power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = -power_X

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEBcat_starved(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables for a starved organism, without accounting for ingestion and assimilation processes, based on the DEB parameters and temperature.
    
    Parameters
    ----------
    t : float
        Time (days)
    y : array
        Array containing the state variables of the organism
    DEB_p : array
        Array containing the DEB parameters of the organism
    temperature : float
        Temperature (K)

    Returns
    -------
    dydt : array
        Array containing the change in state variables of the organism
    '''
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr

    # Growth flux
    power_G = 0

    # Maturity maintenance
    power_J = 0#(k_J*E_H)*arrhenius_corr
    
    # Starved mobilisation (reserve is only used to pay somatic maintenance, no allocation to maturity maintenance, maturation or reproduction)
    power_C = power_S + power_J

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = 0
    power_H = 0

    ### Differences

    dLdt = power_G

    dEdt = (- power_C)

    dE_Hdt = power_H
    
    dE_Rdt = power_R

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = 0

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

@jit(nopython=True)
def DEB_starved(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables for a starved organism and food density state variable based on the DEB parameters and temperature.

    Parameters
    ----------
    t : float
        Time (days).
    y : array
        Array of DEB state variables.
    DEB_p : array
        Array of DEB parameters.
    temperature : float
        Temperature (K).

    Returns
    -------
    dydt : array
        Array of DEB state variable derivatives.
    '''
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    F_m = DEB_p[3]                              # surface-area specific searching rate (cells cm-2 d-1)
    kap_X = DEB_p[4]                            # digestion efficiency (J assimilated/ J ingested)
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    p_T = DEB_p[9]                              # surface-area specific somatic maintenance costs (J cm-2 d-1)
    E_G = DEB_p[11]                             # specific cost of structure (J cm-3)
 
    # Define temperature parameters
    T_A = DEB_p[14]                             # Arrhenius temperature (K)
    T_ref = DEB_p[15]                           # Reference temperature (K)
    
    # Define the aging parameters
    h_a = DEB_p[16]
    s_G = DEB_p[17]
    mu_X = DEB_p[18]
    mu_E = DEB_p[19]
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Feeding conversion
    y_E_X = kap_X * mu_X/mu_E
    y_X_E = 1/y_E_X
    
    K = (p_Am*y_X_E)/F_m # Half saturation constant in terms of energy food
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    
    # Definition functional response & making sure it's zero when food runs out
    if X <= 0:
        X = 0
        
    f = (X/(K+X))

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m

    # Specific growh rate
    L_T = p_T / p_M
    g = E_G/(kap*E_dens_m)
    r_dot = v * (e_scaled / L - (1 + L_T/L)/L_m)/(e_scaled + g)

    ### Determining the powers / energy fluxes based on current state
    
    # Defining the  arrhenius correction
    T_L = DEB_p[20]
    T_H = DEB_p[21]
    TAL = DEB_p[22]
    TAH = DEB_p[23]
    
    gamma_t = (1 + math.exp((TAL/temperature) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/temperature)))
    gamma_t_ref = (1 + math.exp((TAL/T_ref) - (TAL/T_L)) + math.exp((TAH/T_H) - (TAH/T_ref)))

    arrhenius_corr = math.exp((T_A/T_ref)-(T_A/temperature))*gamma_t_ref/gamma_t   

    # Assimilation flux: if the organism is an embryo, it can't feed by itself
    power_A = (p_Am*f*L**2)*arrhenius_corr

    # Feeding flux
    power_X = power_A*y_X_E

    # Somatic maintenance flux
    power_S = (p_M*L**3 + p_T*L**2)*arrhenius_corr
    
    # Maturity maintenance
    power_J = 0
    
    # Growth flux
    power_G = 0
    
    # Mobilisation flux
    power_C = power_S + power_J

    # Reproduction/Maturation flux: if the organism is an adult then the maturity energy flow is directed to reproduction, else it's directed to maturation
    power_R = 0
    power_H = 0
    
    ### Differences

    dLdt = (power_G/(3*(L**2)*E_G))

    dEdt = (power_A - power_C)

    dE_Hdt = (power_H)

    dE_Rdt = (power_R)

    dqdt = ((q*L**3/L_m**3*s_G + h_a)*e_scaled*(v/L - r_dot) - r_dot*q)

    dhdt = (q - r_dot*h)
    
    dXdt = -power_X

    dydt = np.array([dLdt,
            dEdt,
            dE_Hdt,
            dE_Rdt,
            dqdt,
            dhdt,
            dXdt])
    

    return dydt

#%% EVENTS
def eventAttr(terminal, direction):
    '''
    Decorator function to add attributes to the event functions used in the ode solver.

    Parameters
    ----------
    terminal : bool
        Whether the ode solver should stop when the event function is triggered.
    direction : int
        Whether the ode solver should stop when the event function is triggered from above (1), below (-1) or both (0).

    Returns
    -------
    decorator : function
        Decorator function that adds the attributes to the event function.
    '''
    def decorator(func):
        func.terminal = terminal
        func.direction = direction
        return func
    return decorator

@eventAttr(True, 1)
def reach_birth(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches the maturity threshold for birth.

    Parameters
    ----------
    t : float
        Current time.
    y : array
        Current state variables.
    DEB_p : array
        DEB parameters.
    temperature : float
        Current temperature.

    Returns
    -------
    E_H - E_Hb : float
        Difference between maturity and maturity threshold for birth such that the ode solver stops when it reaches 0 or below.
    '''
    # Define DEB parameters
    E_Hb = DEB_p[12]                           # maturity at birth (J)
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    
    return E_H - E_Hb

@eventAttr(True, 1)
def reach_puberty(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches the maturity threshold for puberty.

    Parameters
    ----------
    t : float
        Current time.
    y : array
        Current state variables.
    DEB_p : array
        DEB parameters.
    temperature : float
        Current temperature.

    Returns
    -------
    E_H - E_Hp : float
        Difference between maturity and maturity threshold for puberty such that the ode solver stops when it reaches 0 or below.
    '''
    # Define DEB parameters
    E_Hp = DEB_p[13]                            # maturity at birth (J)
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    
    return E_H - E_Hp

@eventAttr(True, -1)
def reach_starvation(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches starvation conditions, defined as the moment when scaled reserve density e falls below the scaled length l (e < l).
    
    Parameters
    ----------
    t : float
        Current time.
    y : array
        Current state variables.
    DEB_p : array
        DEB parameters.
    temperature : float
        Current temperature.

    Returns
    -------
    e_scaled - l_scaled : float
        Difference between scaled reserve density and scaled length such that the ode solver stops when it reaches 0 or below.
    '''
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m
    l_scaled = L/L_m
    
    return e_scaled-l_scaled

@eventAttr(True, 1)
def reach_normal(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches non-starvation conditions, defined as the moment when scaled reserve density e is equal to or rises above the scaled length l (e => l).

    Parameters
    ----------
    t : float
        Time (days).
    y : array
        State variables.
    DEB_p : array
        DEB parameters.
    temperature : float
        Temperature (°C).

    Returns
    -------
    e_scaled-l_scaled : float
        Difference between scaled reserve density and scaled length, such that the ode solver stops when this difference is equal to or above zero.
    '''
    
    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m
    l_scaled = L/L_m
    
    return e_scaled-l_scaled

@eventAttr(True, -1)
def reach_death(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches reserve-related death conditions, defined as E < 0.

    Parameters
    ----------
    t : float
        Time.
    y : list
        List of state variables.
    DEB_p : list
        List of DEB parameters.
    temperature : float
        Temperature.

    Returns
    -------
    float
        E, to stop the ode solver when E < 0

    '''
    return y[1]

@eventAttr(False, 1)
def empty_event(t, y, DEB_p, temperature):
    '''
    An empty event function that does nothing, to assign to the event attribute of the ode solver when changing the event function.

    Parameters
    ----------
    t : float
        Time.
    y : list
        List of state variables.
    DEB_p : list
        List of DEB parameters.
    temperature : float
        Temperature.

    Returns
    -------
    int
        1, to keep the ode solver running.
    '''
    return 1


#%% Check starvation
def check_starvation(DEB_p, y):
    '''
    Checks if the organism is in starvation conditions, defined as the moment when scaled reserve density e is below the scaled length l (e < l).

    Parameters
    ----------
    DEB_p : list
        List of DEB parameters.
    y : list
        List of state variables.
    
    Returns
    -------
    bool
        True if the organism is in starvation conditions, False otherwise.
    '''

    # Define DEB parameters
    p_Am = DEB_p[2]*DEB_p[1]
    v = DEB_p[5]                                # energy conductance (cm d-1)
    kap = DEB_p[6]                              # allocation fraction to soma (J to growth/ J mobilized)
    p_M = DEB_p[8]                              # vol-specific somatic maintenance costs (J cm-3 d-1)
    
    # Maximum structural length
    L_m = p_Am*kap/p_M
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h = y

    # Scaled reserve density
    E_dens = E/L**3
    E_dens_m = p_Am / v
    e_scaled = E_dens / E_dens_m
    l_scaled = L/L_m
    
    return e_scaled < l_scaled

#%% DEB SOLVERS
def solve_DEB_catabolism(individual, temperature, delta_t, food_density, food_mode):
    
    '''
    Numerically integrates the DEB model (excluding ingestion and assimilation processes) over delta_t, with a given food density and temperature.
    It modifies the individual's state accordingly, taking into account the events that occur during the integration.
    Depending on the food mode, the food density is either the functional response f or the explicit food density X.
    
    Parameters
    ----------
    individual : Individual
        Individual to be integrated.
    temperature : float
        Temperature at which the individual is integrated.
    delta_t : float
        Time interval over which the individual is integrated.
    food_density : float
        Food density at which the individual is integrated.
    food_mode : str
        Food mode, either "f" for functional response or "X" for explicit food density.

    Returns
    -------
    None
    '''

    # Determine the food mode, f for functional response, X for explicity food density
    if food_mode == "f":
        f = food_density
        K = (individual.p_Am*individual.pAm_p_scatter/(individual.kap_X*individual.F_m))
        
        X = f*K/(1.000000001-f)
        
    if food_mode == "X":
        X = food_density
        
    # Define initial conditions
    t_end = delta_t
    t = 0
    y = np.copy(individual.DEB_v)
    y = np.append(y, X)
    
    # Initialize events (if it's an embryo it can reach birth and puberty, if it's a juv it can only reach puberty, if it's an adult these events are not relevant)
    if individual.stage == 1:
        reach_birth_current = reach_birth
        reach_puberty_current = reach_puberty
    elif individual.stage == 2:
        reach_birth_current = empty_event
        reach_puberty_current = reach_puberty
    elif individual.stage == 3:
        reach_birth_current = empty_event
        reach_puberty_current = empty_event
        
    reach_starvation_current = reach_starvation
    reach_normal_current = reach_normal
    reach_death_current = empty_event
    
    if X > 0:
        reach_food_empty_current = reach_food_empty
    elif X == 0:
        reach_food_empty_current = empty_event
    
    # Initialize the correct DEB system
    if individual.starved:
        ode_f = DEBcat_starved
        reach_death_current = reach_death
        
        if individual.stage == 2:
            reach_birth_current = empty_event
        elif individual.stage == 3:
            reach_puberty_current = empty_event
        elif individual.stage == 1:
            ode_f = DEB_normal_em
            
    elif individual.stage == 1:
        ode_f = DEB_normal_em
        
    elif individual.stage == 2:
        ode_f = DEBcat_normal_juv
        reach_birth_current = empty_event
        
    elif individual.stage == 3:
        ode_f = DEBcat_normal_adu
        reach_puberty_current = empty_event
    
    #switched_current = 0
    while True:
        
        sol = solve_ivp(ode_f, [t, t_end], y, args=(individual.DEB_p, temperature), events=(reach_birth_current, reach_puberty_current, reach_starvation_current, reach_normal_current, reach_death_current, reach_food_empty_current), dense_output=True)
        
        if sol.status == 1:
            # if sol.t[0] == sol.t[-1]:
            #     print("stuck")
            # event 1: embryo to juvenile
            if len(sol.y_events[0]) > 0:
                
                # time and state at birth
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # set the right function
                ode_f = DEBcat_normal_juv
                
                # save and update individual life-history
                individual.stage = 2
                individual.lb = y[0]
                individual.ab = individual.age + t
                
                reach_birth_current = empty_event
                
                # break out
                break
                
            # event 2: juvenile to puberty
            elif len(sol.y_events[1]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                ode_f = DEBcat_normal_adu
                individual.stage = 3
                individual.lp = y[0]
                individual.ap = individual.age + t
                
                reach_puberty_current = empty_event
            # event 3: normal to starvation
            elif len(sol.y_events[2]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                ode_f = DEBcat_starved
                individual.starved = True
                
                # Because of the fact that food during the simulation of DEB cannot increase, when an individual starves, it cannot go back to normal within its simulation so both events are shutdown
                reach_starvation_current = empty_event
                reach_normal_current = empty_event
                reach_death_current = reach_death
                    
            # event 4: starvation to normal
            elif len(sol.y_events[3]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                if individual.stage == 2:
                    ode_f = DEBcat_normal_juv
                elif individual.stage == 3:
                    ode_f = DEBcat_normal_adu
                individual.starved = False
                
                # An individual can go from starvation to normal then back to starvation in one simulation, but it can't go back to normal
                reach_normal_current = empty_event
                reach_starvation_current = reach_starvation
                reach_death_current = empty_event
                
            # event 5: starvation to death
            elif len(sol.y_events[4]) > 0:
                individual.dead = True
                individual.lm = y[0]
                individual.am = individual.age + t
                #print("Individual starvation death: " + str(y[1]))
                break
            
            # event 6: food is empty
            elif len(sol.y_events[5]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                y[6] = 0
                reach_food_empty_current = empty_event
        else:
            t = t_end
            break
    
    # store results
    individual.DEB_v = sol.y[:6, -1]
    food_density = sol.y[6, -1]

    # if at the end, reserve is still negative, make em dead
    if individual.E < 0:
        #print("Individual starvation death: " + str(individual.E))
        individual.dead = True
    
    # age up
    individual.age += t
        
    return t

def solve_DEB_total(individual, temperature, delta_t, food_density, food_mode):
    '''
    Numerically integrates the full DEB model over delta_t, with a given food density and temperature.
    It modifies the individual's state accordingly, taking into account the events that occur during the integration.
    Depending on the food mode, the food density is either the functional response f or the explicit food density X.
    
    Parameters
    ----------
    individual : Individual
        The individual to integrate.
    temperature : float
        The temperature at which to integrate.
    delta_t : float
        The time interval over which to integrate.
    food_density : float
        The food density at which to integrate.
    food_mode : string
        The food mode, either "f" for functional response or "X" for explicit food density.

    Returns
    -------
    t : float
        The final time of the integration.
    '''

    y_E_X = individual.kap_X * individual.mu_X/individual.mu_E
    y_X_E = 1/y_E_X
    
    # Determine the food mode, f for functional response, X for explicity food density
    if food_mode == "f":
        f = food_density
        K = (individual.p_Am*individual.pAm_p_scatter*y_X_E/(individual.F_m))
        
        X = f*K/(1.000000001-f)
        
    if food_mode == "X":
        X = food_density
        
    # Define initial conditions
    t_end = delta_t
    t = 0
    y = np.copy(individual.DEB_v)
    y = np.append(y, X)
    
    # Initialize events (if it's an embryo it can reach birth and puberty, if it's a juv it can only reach puberty, if it's an adult these events are not relevant)
    if individual.stage == 1:
        reach_birth_current = reach_birth
        reach_puberty_current = reach_puberty
    elif individual.stage == 2:
        reach_birth_current = empty_event
        reach_puberty_current = reach_puberty
    elif individual.stage == 3:
        reach_birth_current = empty_event
        reach_puberty_current = empty_event
        
    reach_starvation_current = reach_starvation
    reach_normal_current = reach_normal
    
    if X > 0:
        reach_food_empty_current = reach_food_empty
    elif X == 0:
        reach_food_empty_current = empty_event
    
    # Initialize the correct DEB system
    if individual.starved:
        ode_f = DEB_starved
        
        if individual.stage == 2:
            reach_birth_current = empty_event
        elif individual.stage == 3:
            reach_puberty_current = empty_event
        elif individual.stage == 1:
            ode_f = DEB_normal_em
            
    elif individual.stage == 1:
        ode_f = DEB_normal_em
        
    elif individual.stage == 2:
        ode_f = DEB_normal_juv
        reach_birth_current = empty_event
        
    elif individual.stage == 3:
        ode_f = DEB_normal_adu
        reach_puberty_current = empty_event
    
    #switched_current = 0
    while True:
        
        sol = solve_ivp(ode_f, [t, t_end], y, args=(individual.DEB_p, temperature), events=(reach_birth_current, reach_puberty_current, reach_starvation_current, reach_normal_current, reach_death, reach_food_empty_current), dense_output=True)
        
        if sol.status == 1:
            # if sol.t[0] == sol.t[-1]:
            #     print("stuck")
            # event 1: embryo to juvenile
            if len(sol.y_events[0]) > 0:
                
                # time and state at birth
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # set the right function
                ode_f = DEB_normal_juv
                
                # save and update individual life-history
                individual.stage = 2
                individual.lb = y[0]
                individual.ab = individual.age + t
                
                reach_birth_current = empty_event
                
            # event 2: juvenile to puberty
            elif len(sol.y_events[1]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                ode_f = DEB_normal_adu
                individual.stage = 3
                individual.lp = y[0]
                individual.ap = individual.age + t
                
                reach_puberty_current = empty_event
            # event 3: normal to starvation
            elif len(sol.y_events[2]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                ode_f = DEB_starved
                individual.starved = True
                
                # Because of the fact that food during the simulation of DEB cannot increase, when an individual starves, it cannot go back to normal within its simulation so both events are shutdown
                reach_starvation_current = empty_event
                reach_normal_current = empty_event

                    
            # event 4: starvation to normal
            elif len(sol.y_events[3]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                if individual.stage == 2:
                    ode_f = DEB_normal_juv
                elif individual.stage == 3:
                    ode_f = DEB_normal_adu
                individual.starved = False
                
                # An individual can go from starvation to normal then back to starvation in one simulation, but it can't go back to normal
                reach_normal_current = empty_event
                reach_starvation_current = reach_starvation

            # event 5: starvation to death
            elif len(sol.y_events[4]) > 0:
                #print("Starvation Death")
                individual.dead = True
                individual.lm = y[0]
                individual.am = individual.age + t
                break
            
            # event 6: food is empty
            elif len(sol.y_events[5]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                y[6] = 0
                reach_food_empty_current = empty_event
        else:
            t = t_end
            break
    
    # store results
    individual.DEB_v = sol.y[:6, -1]
    #individual.ingested = food_density - sol.y[6,-1]
    food_density = sol.y[6, -1]

    # if at the end, reserve is still negative, make em dead
    if individual.E < 0:
        #print("Starvation Death")
        individual.dead = True
    
    # age up
    individual.age += t
        
    return t

#%%
def init_DEB(individual, stage, development_time):
    '''
    Initializes the DEB system of an individual by simulating it until it reaches the stage specified by the input stage and development time (if the chosen stage is embryo)

    Parameters
    ----------
    individual : Individual
        The individual to initialize
    stage : int
        The stage to initialize the individual to
    development_time : float
        The time to simulate the individual for if the stage is embryo

    Returns
    -------
    None.
    '''

    # Determine initial reserve
    E0 = init_reserve_DEB(individual.DEB_p, 1)
    individual.init_E = E0
    
    # If stage == embryo, then just standard initial conditions
    if  stage == 1:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0], dtype=np.float64))
        individual.age = 0
        
        while individual.age < development_time:
            _ = solve_DEB_total(individual, temperature = 293, delta_t = 1/24, food_density = 1, food_mode = "f")

    # If stage == juvenile, simulate the DEB from embryo to juvenile
    elif stage == 2:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0]))
        individual.age = 0

        while individual.stage < 2:
            _ = solve_DEB_total(individual, temperature = 293, delta_t = 1/24, food_density = 1, food_mode = "f")

            if individual.dead:
                raise Exception("Individual never reaches juvenile stage!")
        
        juv_age = individual.age
        
        while individual.age < (juv_age + development_time):
            _ = solve_DEB_total(individual, temperature = 293, delta_t = 1/24, food_density = 1, food_mode = "f")

    # If stage == adult, simulate the DEB from embryo to adult
    elif stage == 3:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0], dtype=np.float64))
        individual.age = 0

        while individual.stage < 3:
            _ = solve_DEB_total(individual, temperature = 293, delta_t = 1/24, food_density = 1, food_mode = "f")

            if individual.dead:
                raise Exception("Individual never reaches adult stage!")
        
        adu_age = individual.age
        
        while individual.age < (adu_age + development_time):
            _ = solve_DEB_total(individual, temperature = 293, delta_t = 1/24, food_density = 1, food_mode = "f")
            

#%%
def init_genetics(herit, cv, mean_pAm_g_scatter, mean_neutral_g, seed):
    '''
    Initializes the average genotype and the genetic variability of the population for the p_Am trait and a neutral trait

    Parameters
    ----------
    herit : float
        Heritability of the trait
    cv : float
        Coefficient of variation of the trait
    mean_pAm_g_scatter : float
        Genotypic mean of the trait
    mean_neutral_g : float
        Genotypic mean of the neutral trait
    seed : int
        Seed for the random number generator

    Returns
    -------
    gen_p : list
        List of the sampled genotypes for the p_Am and neutral trait
    '''
    rs = np.random.RandomState(seed)
    # unpacking of gen params: genotype & phenotype of heritable traits and genetic variability defined for the lognormally distributed trait
    pheno_var = (cv*mean_pAm_g_scatter)**2 # getting absolute variability in terms of the coefficient of variation
    g_var = herit*pheno_var
    e_var = (1-herit)*pheno_var
    
    # Transformation to normal distribution parameters
    mean_pAm_g_scatter_ln = 2*math.log(mean_pAm_g_scatter)-0.5*math.log(g_var + mean_pAm_g_scatter**2)
    mean_neutral_g_ln = 2*math.log(mean_neutral_g)-0.5*math.log(g_var + mean_neutral_g**2)
    mean_env_ln = 2*math.log(1)-0.5*math.log(e_var + 1**2)
    
    g_var_ln_z= -2*math.log(mean_pAm_g_scatter) + math.log(g_var + mean_pAm_g_scatter**2)
    g_var_ln_neutral = -2*math.log(mean_neutral_g) + math.log(g_var + mean_neutral_g**2)
    e_var_ln = -2*math.log(1) + math.log(e_var + 1**2)
    
    # phenotypic expression according to quantitative genetics theory (using transformed normal distribution)
    pAm_g_scatter_ln = rs.normal(mean_pAm_g_scatter_ln, math.sqrt(g_var_ln_z))
    pAm_p_scatter_ln = pAm_g_scatter_ln + (rs.normal(mean_env_ln, math.sqrt(e_var_ln)))
    
    neutral_g_ln = rs.normal(mean_neutral_g_ln, math.sqrt(g_var_ln_neutral))
    neutral_p_ln = neutral_g_ln + rs.normal(mean_env_ln, math.sqrt(e_var_ln))
    
    # Transforming back to original (lognormal) distribution
    pAm_g_scatter = math.exp(pAm_g_scatter_ln)
    pAm_p_scatter = math.exp(pAm_p_scatter_ln)
    neutral_g = math.exp(neutral_g_ln)
    neutral_p = math.exp(neutral_p_ln)
    
    # making of gen params
    gen_p = np.array([pAm_g_scatter, pAm_p_scatter, neutral_g, neutral_p])
    
    return gen_p

#%%
def checkDeath(individual, delta_t):
    '''
    Checks if the individual dies due to starvation or age.

    Parameters
    ----------
    individual : Individual object
        Individual for which the death is checked
    delta_t : float
        Time step

    Returns
    -------
    None.
    '''    
    # die if aged to death
    r_death = np.random.random()
    if r_death < individual.h*delta_t:
        #print("Individual age death")
        individual.dead = True
        individual.lm = individual.L
        individual.am = individual.age

#%%
#@jit(nopython=True)
def determineNeighbourhood(individual, grid_new):
    '''
    Determines the empty neighbourhood of the individual by returning a list of tuples of the [x, y] coordinates of the empty cells

    Parameters
    ----------
    individual : Individual object
        Individual for which the neighbourhood is determined
    grid_new : numpy array
        Grid on which the individual is located

    Returns
    -------
    empty_neigbourhood : list
        Array of the [x, y] coordinates of the empty cells in the neighbourhood
    '''

    # Determining the positions of the neighbourhood
    grid_length = grid_new.shape[1]
    nbr = np.array([[-1,0], [1,0], [0,1], [0,-1], [1,-1], [1,1], [-1,1], [-1,-1]], dtype=np.int64)  # moore neighbourhood
    nbr_iarray = nbr + individual.current_pos

    # Boundaries
    nbr_iarray = np.where(nbr_iarray > grid_length - 1, 0, nbr_iarray)
    nbr_iarray = np.where(nbr_iarray < 0, grid_length - 1, nbr_iarray)
    
    # Determine the [x, y] coordinates that are empty with numba-friendly advanced indexing
    nbr_grid = np.zeros((nbr_iarray.shape[0]))
    for j in range(nbr_iarray.shape[0]):
        nbr_grid[j] = grid_new[int(nbr_iarray[j, 0]), int(nbr_iarray[j, 1])]
    
    empty_index = np.where(nbr_grid == 0)[0]
    empty_neighbourhood = nbr_iarray[empty_index]
    
    return empty_neighbourhood


#%% 
def determineOffspringParams(individual, herit, iv):
    '''
    Determines the offspring parameters in the individual based on the parent parameters and the heritability and store them in the individual.

    Parameters
    ----------
    individual : Individual object
        The individual that is reproducing.
    herit : float
        The heritability of the trait.
    iv : float
        The individual variability of the trait.

    Returns
    -------
    None.

    '''
    # Function should only be called when offspring bank is empty and individual is mature
    for i in range(int(individual.brood_size)):
        # Inheritance of genotype, DEB parameters and behavioural parameters
        offspring_pAm_g_scatter = individual.pAm_g_scatter
        offspring_neutral_g = individual.neutral_g
        

        # transformation from normal means and variances to lognormal ones
        pheno_var = (iv*offspring_pAm_g_scatter)**2
        g_var = herit*pheno_var
        e_var = (1-herit)*pheno_var
        e_var_ln = -2*math.log(1) + math.log(e_var + 1**2)
        
        mean_pAm_g_scatter_ln = 2*math.log(offspring_pAm_g_scatter)-0.5*math.log(g_var + offspring_pAm_g_scatter**2)
        mean_neutral_g_ln = 2*math.log(offspring_neutral_g)-0.5*math.log(g_var + offspring_neutral_g**2)
        mean_env_ln = 2*math.log(1)-0.5*math.log(e_var + 1**2)
        
        # ensure that a viable parameter set is sampled, check Lika 2011 for parameter ranges
        # phenotypic expression according to quantitative genetics theory
        offspring_pAm_p_scatter_t = mean_pAm_g_scatter_ln + np.random.normal(mean_env_ln, math.sqrt(e_var_ln))
        offspring_neutral_p_t = mean_neutral_g_ln + np.random.normal(mean_env_ln, math.sqrt(e_var_ln))
        
        # transformation back to lognormal distribution
        offspring_pAm_p_scatter = math.exp(offspring_pAm_p_scatter_t)
        offspring_neutral_p = math.exp(offspring_neutral_p_t)
        
        # Insertion in DEB parameters
        offspring_DEB_p = np.copy(individual.DEB_p)
        offspring_DEB_p[0] = offspring_pAm_g_scatter
        offspring_DEB_p[1] = offspring_pAm_p_scatter
        
        # Store parameters in offspring_bank
        individual.offspring_bank[i, :] = np.array([offspring_pAm_g_scatter, offspring_pAm_p_scatter, offspring_neutral_g, offspring_neutral_p, -1])
        
        # initialize threshold as negative number to signal that it should be calculated afterwards accounting for reserve density
        individual.rep_threshold = -1

#%%
def determineRepThreshold(individual, herit, iv):
    '''
    Sets the reproduction threshold of the individual based on the offspring parameters and the heritability that are already stored in the individual.

    Parameters
    ----------
    individual : individual object
        The individual for which the reproduction threshold should be determined.
    herit : float
        Heritability (0-1).
    iv : float
        Individual variability.

    Returns
    -------
    None.
    '''

    # function should only be called when offspring bank is NOT empty and individual is mature
    # reset threshold and bank
    individual.rep_threshold = 0
    #individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
    
    # recalculate for every individual in brood size
    for i in range(int(individual.brood_size)):
        
        # Determine mothers scaled reserve density
        E_dens = individual.E/individual.L**3
        E_dens_m = individual.p_Am*individual.pAm_p_scatter/individual.v
        eb = E_dens/E_dens_m
        
        # Recall the offspring parameters
        offspring_DEB_p = np.copy(individual.DEB_p)
        offspring_DEB_p[0] = individual.offspring_bank[i, 0]
        offspring_DEB_p[1] = individual.offspring_bank[i, 1]
        
        # Determine initial reserve necessary
        E_0 = init_reserve_DEB(offspring_DEB_p, eb)
        
        # Store parameters in offspring_bank
        individual.offspring_bank[i, 4] = E_0/individual.kap_R
        
        # Adding its initial reserve to the threshold
        individual.rep_threshold += E_0/individual.kap_R
    
#%%
### Move and reproduce functions ###
def move(grid, individual, empty_neighbourhood, movement_rule=0):
    ''' 
    Moves the individual on the grid to a random empty spot in its neighbourhood (random walk).

    Parameters
    ----------
    grid : 2d numpy array
        The grid on which the individuals are located
    individual : individual object
        The individual that is moving
    empty_neighbourhood : list of tuples
        List of tuples containing the coordinates of the empty cells in the neighbourhood of the individual
    movement_rule : int, optional
        The movement rule that is used to determine the weights of the different possible positions. The default is 0 (random walk).
    
    Returns
    -------
    None
    '''
    # Remove from beginning position in grid
    remove_individual_from_grid(individual, grid)
    
    # Determine the weights of the different possible positions based on the rule
    # Rule 0: random walk
    # Rule 1: biased random walk positively skewed to food
    # Rule 2: maximum neighbourhood food
    if movement_rule == 0:
        weights = np.ones(len(empty_neighbourhood))/len(empty_neighbourhood)
        final_pos = empty_neighbourhood[np.random.choice(len(empty_neighbourhood), 1, p = weights)[0], :]
        
    if movement_rule == 1:
        total_food_in_neighbourhood = np.sum(grid['food'][empty_neighbourhood[:,0], empty_neighbourhood[:,1]])
        food_per_patch = grid['food'][empty_neighbourhood[:,0], empty_neighbourhood[:,1]]
        
        if total_food_in_neighbourhood == 0:
            weights = np.ones(len(empty_neighbourhood))/len(empty_neighbourhood)
        else:
            weights = food_per_patch/total_food_in_neighbourhood
        
        final_pos = empty_neighbourhood[np.random.choice(len(empty_neighbourhood), 1, p = weights)[0], :]
    if movement_rule == 2:
        food_per_patch = grid['food'][empty_neighbourhood[:,0], empty_neighbourhood[:,1]]
        final_pos = empty_neighbourhood[np.random.choice(np.where(food_per_patch == np.max(food_per_patch))[0]),:]
        
    # update the position attribute of this instance
    individual.current_pos = np.copy(final_pos)
    
    # reset the within-cell movement effort
    individual.movement = 0
    
    # Update the movement on the grid
    put_individual_in_grid(individual, grid)

def reproduce(grid, individual, population, empty_neighbourhood, max_id):
    '''
    Handles the reproduction of an individual by modifying the population list and the grid.

    Parameters
    ----------
    grid : 2D matrix
        The grid on which the individuals are located.
    individual : Individual object
        The individual that is reproducing.
    population : list
        The list of all individuals in the population.
    empty_neighbourhood : list
        The list of empty cells in the neighbourhood of the individual.
    max_id : int
        The highest id of the individuals in the population.

    Returns
    -------
    population : list
        The updated list of all individuals in the population.
    max_id : int
        The updated highest id of the individuals in the population.

    '''
    # shuffle up the neihgbourhood
    nbr = np.copy(empty_neighbourhood)
    np.random.shuffle(nbr)
    
    if len(nbr) < individual.brood_size:
        # remove egg cost from buffer
        individual.R -= individual.offspring_bank[0, 4]
        
        # add to the potential fecundity
        individual.potential_fecundity += 1
        # egg failed though so no other action
    else:
        # reproduce as many offspring as your brood_size
        # pick a random spot
        r_spot = nbr[0]
        
        # make an embryo with the offspring parameters
        offspring_params = individual.offspring_bank[0,:]
        
        # inherit the genetic traits
        offspring_gen_p = np.array([0., 0., 0., 0.], dtype=np.float64)
        offspring_gen_p[0] = offspring_params[0] # pAm_g_scatter
        offspring_gen_p[1] = offspring_params[1] # pAm_p_scatter
        offspring_gen_p[2] = offspring_params[2] # neutral_g
        offspring_gen_p[3] = offspring_params[3] # neutral_p
        
        # inherit the deb parameters (with appropriate heritable traits)
        offspring_DEB_p = np.copy(individual.DEB_p)
        offspring_DEB_p[0] = offspring_gen_p[0] #pAm_g_scatter
        offspring_DEB_p[1] = offspring_gen_p[1] #pAm_p_scatter
        
        # inherit the behaviour parameters
        offspring_behave_p = individual.behave_p
        
        # determine the position where the new embryo will be deposited
        pos = np.copy(r_spot)
        
        # determine the generation of the offspring
        generation_offspring = individual.generation + 1
        max_id += 1
        # make the individual
        offspring = Individual(max_id, offspring_DEB_p, offspring_behave_p, offspring_gen_p, 1, pos, generation_offspring)
        
        # initialise it
        offspring.L = 10**-8
        offspring.E = offspring_params[4]*individual.kap_R #E_0
        offspring.init_E = offspring_params[4]*individual.kap_R
        offspring.age = 0
        offspring.stage = 1
        
        # Add offspring to grid and population list
        population = np.append(population, offspring)
        put_individual_in_grid(offspring, grid)
        
        # remove egg cost from buffer
        individual.R -= individual.offspring_bank[0, 4]
        
        # add to the fecundity counter of the individual
        individual.fecundity += 1
        individual.potential_fecundity += 1
        
        # Update the individual in the grid
        put_individual_in_grid(individual, grid)
     
    return population, max_id

### Update environment ###
@jit(nopython=True)
def updateEnvironmentDailyConstantAddition(grid, food_capacity, food_diffusion_rate, temperature, delta_t, cell_length, t):
    '''
    Updates the environment by adding a constant amount of food to the grid and diffusing the food. The grid is updated in place. Used in the simulation of the case study.

    Parameters
    ----------
    grid : 2D matrix
        The grid on which the individuals are located.
    food_capacity : float
        The amount of food that is added to the grid.
    food_diffusion_rate : float
        The diffusion rate of the food.
    temperature : float
        The temperature of the environment.
    delta_t : float
        The time step.
    cell_length : float
        The length of a cell.
    t : float
        The current time.

    Returns
    -------
    None.
    '''    
    if round(t, 4) % 1 == 0:
        grid['food'][:, :] += food_capacity
        
    grid['temperature'][:, :] = temperature
    
    
    # Diffusion
    dx2 = 1 # one cell length
    dy2 = 1 # one cell length
    D = (food_diffusion_rate/cell_length**2)# expressed in cell area/day
    dt = (dx2*dy2)/(4*D)
    
    u_init = grid['food'][:, :]
    u_ext = np.zeros((u_init.shape[0]+2, u_init.shape[1]+2))
    
    # Inside
    u_ext[1:-1, 1:-1] = u_init
    
    # Left and right border
    u_ext[1:-1, 0] = u_init[:, -1]
    u_ext[1:-1, -1] = u_init[:, 0]
    
    # Up and bottom border
    u_ext[0, 1:-1] = u_init[-1, :]
    u_ext[-1, 1:-1] = u_init[0, :]
    
    # Initialize variables for loop 
    u_ext0 = np.copy(u_ext)
    t_in = np.arange(0, delta_t, dt)
    
    # if the IBM step is smaller than the stability criterium for stable timesteps, pick the IBM timestep
    if delta_t < dt:
        dt = delta_t
        
    # Diffusion
    for m in range(len(t_in)):
        u_ext[1:-1, 1:-1] = u_ext0[1:-1, 1:-1] + D * dt * ((u_ext0[2:, 1:-1] - 2*u_ext0[1:-1, 1:-1] + u_ext0[:-2, 1:-1])/dx2 + (u_ext0[1:-1, 2:] - 2*u_ext0[1:-1, 1:-1] + u_ext0[1:-1, :-2])/dy2 )
        u_ext0 = np.copy(u_ext)
    
    grid['food'][:, :] = np.copy(u_ext[1:-1, 1:-1])


@jit(nopython=True)
def updateEnvironmentLogisticGrowth(grid, food_capacity, food_growth_rate, food_diffusion_rate, temperature, delta_t, cell_length, t):
    '''
    Updates the environment by adding food to the grid according to logistic growth and diffusing the food. The grid is updated in place. Used in the simulation of the simulation study.

    Parameters
    ----------
    grid : 2D matrix
        The grid on which the individuals are located.
    food_capacity : float
        The carrying capacity of the food.
    food_growth_rate : float
        The growth rate of the food.
    food_diffusion_rate : float
        The diffusion rate of the food.
    temperature : float
        The temperature of the environment.
    delta_t : float
        The time step.
    cell_length : float
        The length of a cell.
    t : float
        The current time.

    Returns
    -------
    None.

    '''  
    # Setting the Euler step size
    step_size = 1/24/60 #1 minute
    t_in = np.arange(0, delta_t, step_size)
    
    # Defining initial conditions
    previous_state = grid['food'][:, :]
    
    # Solve euler over one day
    for i in range(0, len(t_in)):
        next_state = previous_state + (food_growth_rate*previous_state*(1-(previous_state/food_capacity)))*step_size
        previous_state = next_state
    
    grid['food'][:, :] = next_state
    grid['temperature'][:, :] = temperature
    
    
    # Diffusion
    dx2 = 1 # one cell length
    dy2 = 1 # one cell length
    D = (food_diffusion_rate/cell_length**2)# expressed in cell area/day
    dt = (dx2*dy2)/(4*D)
    
    u_init = grid['food'][:, :]
    u_ext = np.zeros((u_init.shape[0]+2, u_init.shape[1]+2))
    
    # Inside
    u_ext[1:-1, 1:-1] = u_init
    
    # Left and right border
    u_ext[1:-1, 0] = u_init[:, -1]
    u_ext[1:-1, -1] = u_init[:, 0]
    
    # Up and bottom border
    u_ext[0, 1:-1] = u_init[-1, :]
    u_ext[-1, 1:-1] = u_init[0, :]
    
    # Initialize variables for loop 
    u_ext0 = np.copy(u_ext)
    t_in = np.arange(0, delta_t, dt)

    # if the IBM step is smaller than the stability criterium for stable timesteps, pick the IBM timestep
    if delta_t < dt:
        dt = delta_t    

    # Diffusion
    for m in range(len(t_in)):
        u_ext[1:-1, 1:-1] = u_ext0[1:-1, 1:-1] + D * dt * ((u_ext0[2:, 1:-1] - 2*u_ext0[1:-1, 1:-1] + u_ext0[:-2, 1:-1])/dx2 + (u_ext0[1:-1, 2:] - 2*u_ext0[1:-1, 1:-1] + u_ext0[1:-1, :-2])/dy2 )
        u_ext0 = np.copy(u_ext)
    
    grid['food'][:, :] = np.copy(u_ext[1:-1, 1:-1])


#%% Forage functions

def forageAsync(individual, grid_new, delta_t, mr):
    '''
    Handles the foraging of an individual in an asynchronous schedule.

    Parameters
    ----------
    individual : Individual object
        The individual that is foraging.
    grid_new : 2d numpy array
        2d array containing the new grid.
    delta_t : float
        The time step of the model.
    mr : float
        The movement rule of the individual.

    Returns
    -------
    grid_new : 2d numpy array
        updated 2d array containing the new grid.

    '''
    # Define some key parameters
    y_E_X = individual.kap_X * individual.mu_X/individual.mu_E # yield of reserve on food
    y_X_E = 1/y_E_X # yield of food on reserve
    temperature = grid_new["temperature"][individual.current_pos[0], individual.current_pos[1]]
    
    gamma_t = (1 + math.exp((individual.T_AL/temperature) - (individual.T_AL/individual.T_L)) + math.exp((individual.T_AH/individual.T_H) - (individual.T_AH/temperature)))
    gamma_t_ref = (1 + math.exp((individual.T_AL/individual.T_ref) - (individual.T_AL/individual.T_L)) + math.exp((individual.T_AH/individual.T_H) - (individual.T_AH/individual.T_ref)))
    
    
    arrhenius_corr = math.exp((individual.T_A/individual.T_ref)-(individual.T_A/temperature))*gamma_t_ref/gamma_t 
    F_m_corr = individual.F_m*arrhenius_corr
    p_Am_corr = individual.p_Am*individual.pAm_p_scatter*arrhenius_corr
    
    
    # Foraging algorithm
    time_spent_foraging = 0
    total_ingestion = 0
    total_time_spent_searching = 0
    total_time_spent_handling = 0
    while time_spent_foraging < delta_t:
        # as long the individual is not in a cell with food it will move around till it finds food or can't move
        time_spent_searching_current_cycle = 0
        
        # calculate individual speed
        speed = F_m_corr * individual.L**2 # cells per day
        time_spent_searching_per_cell = 1/speed #cells divided by cells per day = days
        while grid_new["food"][individual.current_pos[0], individual.current_pos[1]] == 0:
            # determine empty neighbourhood
            empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
            
            # if empty neighbourhood is null then proceed to the handling
            if len(empty_neighbourhood) == 0:
                break
            # if the time it needs to move exceeds timestep, it puts its remaining available effort into moving to another cell, but doesn't move
            elif (time_spent_searching_per_cell + time_spent_foraging) > delta_t:
                individual.movement += (delta_t - time_spent_foraging)*speed # distance in cells and should be less than 1
                time_spent_searching_current_cycle += (delta_t - time_spent_foraging)
                
                # if the sum of the extra available efforts exceed the effort needed to move 1 cell, move 1 cell
                if individual.movement >= 1:
                    move(grid_new, individual, empty_neighbourhood, movement_rule = mr)
                    individual.movement += 1 - individual.movement
                    
                # end the foraging cycle
                time_spent_foraging = delta_t

                break
            else:
                # otherwise move
                move(grid_new, individual, empty_neighbourhood, movement_rule = mr)
        
                time_spent_foraging += time_spent_searching_per_cell
                time_spent_searching_current_cycle += time_spent_searching_per_cell
                
                # if time spent on foraging reaches delta_t during the searching then stop searching and move on
                if time_spent_foraging >= delta_t:
                    break
        
        total_time_spent_searching += time_spent_searching_current_cycle
        # if time spent on foraging after searching takes up all the available time in delta t, move on and simulate deb
        if time_spent_foraging >= delta_t:
            # break out of the foraging loop
            break
        
        time_spent_handling_current_cycle = 0
        # how much does it instantaneously eat over a remaining handling period delta_t-time_spent_foraging
        max_ingestion = p_Am_corr*y_X_E*(delta_t-time_spent_foraging)*individual.L**2 #Max ingestion in J food
        food = grid_new["food"][individual.current_pos[0], individual.current_pos[1]]
        
        # add maximum ingestion rate if possible otherwise add whatever is there
        if (food - max_ingestion <= 0):
            ingestion = food
        else:
            ingestion = max_ingestion
            time_spent_foraging = delta_t
        
        individual.E += ingestion*y_E_X
        food_updated = food - ingestion
        total_ingestion += ingestion
        
        # time spent handling food
        time_spent_handling_current_cycle = ingestion*y_E_X/(p_Am_corr*individual.L**2)
        
        # if the neighbourhood is filled then the individual will spend all its time handling food
        empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
        if len(empty_neighbourhood) == 0:
            time_spent_handling_current_cycle = delta_t - time_spent_foraging
        
        total_time_spent_handling += time_spent_handling_current_cycle
        # # forward time with time spent handling and searching of current cycle
        # step_size = time_spent_handling + time_spent_searching_current_cycle
        # temperature = grid_new["temperature"][individual.current_pos[0], individual.current_pos[1]]
        # _ = solve_DEB_catabolism(individual, temperature, step_size, 1, food_mode = "X")
        
        # update grid and total time spent foraging
        #put_individual_in_grid(individual, grid_new)
        grid_new["food"][individual.current_pos[0], individual.current_pos[1]] = food_updated
        time_spent_foraging += time_spent_handling_current_cycle
    
    # determine whether the organism is starving or not after having foraged
    if check_starvation(individual.DEB_p, individual.DEB_v):
        individual.starved = True
    else:
        individual.starved = False
    
    # calculate the functional response in this foraging cycle
    max_total_ingestion = p_Am_corr*y_X_E*(delta_t)*individual.L**2 
    individual.f_resp_5min = total_ingestion/max_total_ingestion

    #forward DEB state variables in time
    temperature = grid_new["temperature"][individual.current_pos[0], individual.current_pos[1]]
    _ = solve_DEB_catabolism(individual, temperature, delta_t, 1, food_mode = "X")
    individual.time_spent_searching = total_time_spent_searching
    individual.time_spent_handling = total_time_spent_handling
    individual.ingested += total_ingestion
    
    # OPTIONAL: neighbourhood check
    empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
    if 8 - len(empty_neighbourhood) > 0:
        individual.neighbours += 1
    
    put_individual_in_grid(individual, grid_new)
        
    return grid_new

    
#%% UPDATE FUNCTIONS

def individualForaging(individual, grid_new, delta_t):
    '''
    General foraging function for individuals.

    Parameters
    ----------
    individual : Individual object
        Individual that is foraging.
    grid_new : 2d numpy array
        Grid that the individual is foraging on.
    delta_t : float
        Time step.

    Returns
    -------
    individual : Individual object
        Updated individual that is foraging.
    grid_new : 2d numpy array
        Updated grid that the individual is foraging on.

    '''
    # Define the row and column of the individual
    row = individual.current_pos[0]
    col = individual.current_pos[1]
    
    # Get environmental parameters
    temperature = grid_new['temperature'][row, col]

    # Foraging and growth of embryo
    t_forage = 0
    
    # if individual is embryo, simulate DEB until its birth then let it forage
    while t_forage < delta_t:
        if individual.stage > 1:
            grid_new = forageAsync(individual, grid_new, (delta_t - t_forage))
            t_forage = delta_t
        else:
            t_embryo = solve_DEB_catabolism(individual, temperature, delta_t, 1, food_mode = "X")
            put_individual_in_grid(individual, grid_new)
            t_forage = t_embryo
    
    return individual, grid_new

def individualAgeing(individual, grid_new, population, index, delta_t):
    '''
    General ageing function for individuals.

    Parameters
    ----------
    individual : Individual object
        Individual that is ageing.
    grid_new : 2d numpy array
        Grid that the individual is ageing on.
    population : list
        List of all individuals in the population.
    index : int
        Index of the individual in the population list.
    delta_t : float
        Time step.

    Returns
    -------
    individual : Individual object
        Updated individual that is ageing.
    grid_new : 2d numpy array
        Updated grid that the individual is ageing on.
    population : list
        Updated list of all individuals in the population.

    '''
    # Check for individual death (starvation & ageing)
    checkDeath(individual, delta_t)
    if individual.dead:
        remove_individual_from_grid(individual, grid_new) # remove individual data from the grid
        population = np.delete(population, index) # remove from individual from the population list
    
    return individual, grid_new, population
    

def IBMschedule(population, grid_new, delta_t, herit, iv, max_id, mr):
    '''
    Main scheduling function for the individuals in the IBM.

    Parameters
    ----------
    population : list
        List of all individuals in the population.
    grid_new : 2d numpy array
        Grid that the individuals are on.
    delta_t : float
        Time step.
    herit : float
        Heritability of the variability.
    iv : float
        Total individual variability
    max_id : int
        Maximum id of the individuals in the population.
    mr : int
        Movement rule.

    Returns
    -------
    population : list
        Updated list of all individuals in the population.
    grid_new : 2d numpy array
        Updated grid that the individuals are on.
    max_id : int
        Updated maximum id of the individuals in the population.
    '''
    index = 0
    for individual in population:
        
        # Define the row and column of the individual
        row = individual.current_pos[0]
        col = individual.current_pos[1]
        
        # Get environmental parameters
        temperature = grid_new['temperature'][row, col]
    
        # Foraging and growth of embryo
        t_forage = 0
        
        # Determine movement rule depending on zone of influence if that is signalled
        if mr == 3:
            occupant_list = determineZoneOfInfluence(individual, grid_new, population, delta_t)
            if len(occupant_list) > 0:
                mr_temp = 0 # random
            else:
                mr_temp = 1 # biased
        else:
            mr_temp = mr
        
        # if individual is embryo, simulate DEB until its birth then let it forage
        while t_forage < delta_t:
            if individual.stage > 1:
                grid_new = forageAsync(individual, grid_new, (delta_t - t_forage), mr_temp)
                t_forage = delta_t
            else:
                t_embryo = solve_DEB_catabolism(individual, temperature, delta_t, 1, food_mode = "X")
                put_individual_in_grid(individual, grid_new)
                t_forage = t_embryo
        
        # Check for individual death (starvation & ageing)
        checkDeath(individual, delta_t)
        if individual.dead:
            remove_individual_from_grid(individual, grid_new) # remove individual data from the grid
            population = np.delete(population, index) # remove from individual from the population list
            continue
        
        # Move forward the index of the for loop
        index += 1
        
        # Determine empty spots (keep track of where the individual is)
        empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
        #individual.neighbours = 8 - len(empty_neighbourhood)
        
        # Inheritance and offspring params if adult and empty bank or impossible parameterset (rep threshold is zero)
        if (individual.stage == 3) & (individual.rep_threshold == 0):
            determineOffspringParams(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
            
        # Determine reproduction threshold if adult and offspring params ready in bank (rep threshold is -1)
        if (individual.stage == 3) & (individual.rep_threshold < 0):
            determineRepThreshold(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
        
        # reproduce if enough buffer and have viable eggs
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold > 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduce(grid_new, individual, population, empty_neighbourhood, max_id)
                # keep checking neighbourhood after each reproduction event
                empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
            continue
        
        
        # store neighbourhood
    return population, grid_new, max_id

def updateAsync(grid, population, t, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling="random"):
    '''
    General update function in the IBM following the asynchronous update rule. Uses the discrete environment update function for the case study.

    Parameters
    ----------
    grid : 2d numpy array
        Grid that the individuals are on.
    population : list
        List of all individuals in the population.
    t : int
        Current time step.
    delta_t : float
        Time step.
    temperature : 2d numpy array
        Temperature grid.
    food_capacity : 2d numpy array
        Food capacity grid.
    food_diffusion_rate : 2d numpy array
        Food diffusion rate grid.
    herit : float
        Heritability of the variability.
    iv : float
        Total individual variability
    cell_length : float
        Length of a cell.
    max_id : int
        Maximum id of the individuals in the population.
    mr : int
        Movement rule.
    scheduling : str, optional
        Scheduling rule. The default is "random".

    Returns
    -------
    grid_new : 2d numpy array
        Updated grid that the individuals are on.
    population : list
        Updated list of all individuals in the population.
    max_id : int
        Updated maximum id of the individuals in the population.
    

    '''
        
    # New Update
    grid_new = np.copy(grid)
     
    # Scheduling (make function out of this)
    if scheduling == "random":
        np.random.shuffle(population)
    elif scheduling == "size":
        population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    elif scheduling == "stage":
        population = sorted(list(population), key=lambda x: x.stage, reverse=True) # stage priority (adult > juvenile) scheduling
    
    # Individuals - Environment - Individuals update
    updateEnvironmentDailyConstantAddition(grid_new, food_capacity, food_diffusion_rate, temperature, delta_t/2, cell_length, t)
    
    population, grid_new, max_id = IBMschedule(population, grid_new, delta_t, herit, iv, max_id, mr)
    
    updateEnvironmentDailyConstantAddition(grid_new, food_capacity, food_diffusion_rate, temperature, delta_t/2, cell_length, t + delta_t/2)
    
    
    return [grid_new, population, max_id]


def updateAsyncLogisticGrowth(grid, population, t, delta_t, temperature, food_capacity, food_growth_rate, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling="random"):
    '''
    General update function in the IBM following the asynchronous update rule. Uses the logistic growth environment update function for the simulation study.

    Parameters
    ----------
    grid : 2d numpy array
        Grid that the individuals are on.
    population : list
        List of all individuals in the population.
    t : int
        Current time step.
    delta_t : float
        Time step.
    temperature : 2d numpy array
        Temperature grid.
    food_capacity : 2d numpy array
        Food capacity grid.
    food_growth_rate : 2d numpy array
        Food growth rate grid.
    food_diffusion_rate : 2d numpy array
        Food diffusion rate grid.
    herit : float
        Heritability of the variability.
    iv : float
        Total individual variability
    cell_length : float
        Length of a cell.
    max_id : int
        Maximum id of the individuals in the population.
    mr : int
        Movement rule.
    scheduling : str, optional
        Scheduling rule. The default is "random".

    Returns
    -------
    grid_new : 2d numpy array
        Updated grid that the individuals are on.
    population : list
        Updated list of all individuals in the population.
    max_id : int
        Updated maximum id of the individuals in the population.
    '''
        
    # New Update
    grid_new = np.copy(grid)
     
    # Scheduling (make function out of this)
    if scheduling == "random":
        np.random.shuffle(population)
    elif scheduling == "size":
        population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Individuals - Environment - Individuals update
    updateEnvironmentLogisticGrowth(grid_new, food_capacity, food_growth_rate, food_diffusion_rate, temperature, delta_t/2, cell_length, t)
    
    population, grid_new, max_id = IBMschedule(population, grid_new, delta_t, herit, iv, max_id, mr)
    
    updateEnvironmentLogisticGrowth(grid_new, food_capacity, food_growth_rate, food_diffusion_rate, temperature, delta_t/2, cell_length, t + delta_t/2)
    
    
    return [grid_new, population, max_id]


#%% RUN FUNCTIONS

def resetFecundityIngestion(population):
    '''
    Helper function to reset fecundity, ingestion, and neighbours of all individuals in the population.

    Parameters
    ----------
    population : list
        List of individuals.

    Returns
    -------
    None.
    '''
    for individual in population:
        individual.potential_fecundity = 0
        individual.fecundity = 0
        individual.ingested = 0
        individual.neighbours = 0

def runObserveEveryDayAsync(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling, with_tqdm=False):
    '''
    Runs the model for total_time with delta_t time steps. Observe the population size and food every day and record the individual response every day. Environment is updated daily in discrete steps as in the case study.

    Parameters
    ----------
    init_matrix : numpy.ndarray
        Initial grid.
    init_population : list
        Initial population.
    df : pandas.DataFrame
        Dataframe to record individual response.
    total_time : int
        Total time to run the model.
    delta_t : float
        Time step.
    temperature : numpy.ndarray
        Temperature time series.
    food_capacity : float
        Food capacity of the grid.
    food_diffusion_rate : float
        Food diffusion rate.
    herit : float
        Heritability of the trait.
    iv : float
        Total individual variability
    cell_length : float
        Length of the grid cell.
    max_id : int
        Maximum id of the individuals.
    mr : int
        movement rule.
    scheduling : int
        Scheduling rule.
    with_tqdm : bool, optional
        Whether to show progress bar. The default is False.

    Returns
    -------
    grids : numpy.ndarray
        Grids of the simulation.
    popu : numpy.ndarray
        Population sizes of the simulation.
    df : pandas.DataFrame
        Dataframe recording individual responses.
    ts : numpy.ndarray
        Time series.
    '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((total_time + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([total_time + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    if with_tqdm:
        for t in tqdm(ts[1:]):
        
            # observe food and population size at start of update
            if round(t, 4) % 1 == 0:
                grids[int(round(t))] = np.copy(grid_current)
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
            
            # update grid
            grid_new, population_new, max_id = updateAsync(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
                # reset fecundity
                resetFecundityIngestion(population_new)
                
            index += 1
            
            # replace current grid with new grid
            grid_current = np.copy(grid_new)
            population_current = np.copy(population_new)
    else:
    
        for t in (ts[1:]):
            
            # observe food and population size at start of update
            if round(t, 4) % 1 == 0:
                grids[int(round(t))] = np.copy(grid_current)
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
            
            # update grid
            grid_new, population_new, max_id = updateAsync(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
                # reset fecundity
                resetFecundityIngestion(population_new)
                
            index += 1
            
            # replace current grid with new grid
            grid_current = np.copy(grid_new)
            population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def runObserveEveryDayAsyncLogisticGrowth(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_growth_rate, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling, with_tqdm=False):
    '''
    Runs the model for total_time with delta_t time steps. Observe the population size and food every day and record the individual response every day. Environment is updated logistically as in the simulation study.

    Parameters
    ----------
    init_matrix : numpy.ndarray
        Initial grid.
    init_population : list
        Initial population.
    df : pandas.DataFrame
        Dataframe to record individual response.
    total_time : int
        Total time to run the model.
    delta_t : float
        Time step.
    temperature : numpy.ndarray
        Temperature time series.
    food_capacity : float
        Food capacity of the grid.
    food_growth_rate : float
        Food growth rate.
    food_diffusion_rate : float
        Food diffusion rate.
    herit : float
        Heritability of the trait.
    iv : float
        Total individual variability
    cell_length : float
        Length of the grid cell.
    max_id : int
        Maximum id of the individuals.
    mr : int
        movement rule.
    scheduling : int
        Scheduling rule.
    with_tqdm : bool, optional
        Whether to show progress bar. The default is False.

    Returns
    -------
    grids : numpy.ndarray
        Grids of the simulation.
    popu : numpy.ndarray
        Population sizes of the simulation.
    df : pandas.DataFrame
        Dataframe recording individual responses.
    ts : numpy.ndarray
        Time series.
    '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((total_time + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([total_time + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    if with_tqdm:
        for t in tqdm(ts[1:]):
        
            # observe food and population size at start of update
            if round(t, 4) % 1 == 0:
                grids[int(round(t))] = np.copy(grid_current)
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
            
            # update grid
            grid_new, population_new, max_id = updateAsyncLogisticGrowth(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_growth_rate, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
                # reset fecundity
                resetFecundityIngestion(population_new)
                
            index += 1
            
            # replace current grid with new grid
            grid_current = np.copy(grid_new)
            population_current = np.copy(population_new)
    else:
    
        for t in (ts[1:]):
            
            # observe food and population size at start of update
            if round(t, 4) % 1 == 0:
                grids[int(round(t))] = np.copy(grid_current)
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
            
            # update grid
            grid_new, population_new, max_id = updateAsyncLogisticGrowth(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_growth_rate, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
                # reset fecundity
                resetFecundityIngestion(population_new)
                
            index += 1
            
            # replace current grid with new grid
            grid_current = np.copy(grid_new)
            population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

### Function that removes an individual from output grid
def remove_individual_from_grid(individual, grid):
    '''
    Helper function that removes an individual from output grid

    Parameters
    ----------
    individual : Individual object
        Individual to be removed from grid.
    grid : numpy array
        Grid to be updated.

    Returns
    -------
    None.
    '''
    # general organism state
    grid['occupant'][individual.current_pos[0], individual.current_pos[1]] = 0
    
### Function that puts an individual in output grid
def put_individual_in_grid(individual, grid):
    '''
    Helper function that puts an individual in output grid

    Parameters
    ----------
    individual : Individual object
        Individual to be put in grid.
    grid : numpy array
        Grid to be updated.

    Returns
    -------
    None.
    '''
    # general organism state
    grid['occupant'][individual.current_pos[0], individual.current_pos[1]] = individual.stage
    
### Construction of initial matrix ###

def construct_init_matrix(init_individuals, grid_length, DEB_p, gen_init_params, behave_p, init_temperature, init_food_density, seed=None, max_id=0):
    '''
    Function that constructs the initial matrix for the simulation. Initializes the embryos, juveniles and adults in the grid.

    Parameters
    ----------
    init_individuals : list
        List of initial number of individuals in the grid. [N_em, N_juv, N_adu]
    grid_length : int
        Length of the grid.
    DEB_p : list
        List of DEB parameters.
    gen_init_params : list
        List of initial parameters for the genetics of individuals.
    behave_p : list
        List of behavioural parameters.
    init_temperature : float
        Initial temperature of the grid.
    init_food_density : float
        Initial food density of the grid.
    seed : int, optional
        Seed for random number generator. The default is None.
    max_id : int, optional
        Maximum id of individuals. The default is 0.

    Returns
    -------
    init_matrix : numpy array
        Initial matrix of the grid.
    init_population : numpy array
        Initial population of the grid.
    df : pandas dataframe
        Dataframe to store the data of the individuals.

    '''
    # define custom numpy dtype
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    
    # Initialize grid
    init_matrix = np.zeros((grid_length, grid_length), dtype=patchdt)
    init_population = np.zeros(np.sum(init_individuals), dtype=object)
    df = pd.DataFrame({'timestep': [], 'id': [], 'age': [], 'stage': [], 'L': [], 'E': [], 'R': [], 'fecundity': [], 'potential_fecundity': [], 'H': [], 'E_0': [], 'starved': [], 'generation': [], 'time_spent_handling': [], 'time_spent_searching': [], 'ingested': [], 'f_resp_5min': [], 'p_Am': [], 'x_pos': [], 'y_pos': [], 'neighbours': []})
    
    init_matrix['food'] = init_food_density
    init_matrix['temperature'] = init_temperature
    
    # Define number of initial population
    N_em = init_individuals[0]
    N_juv = init_individuals[1]
    N_ad = init_individuals[2]
    N = N_em + N_juv + N_ad
    
    #fix seed if seed given
    if seed:
        rs = np.random.RandomState(seed)
        seeds = rs.choice(999999, size=N, replace=False)
    else:
        seeds = np.random.choice(999999, size=N, replace=False)

    
    # Determining random positions in the grid & making sure they're unique
    all_pos = np.indices((grid_length, grid_length)).reshape((2,grid_length**2)).transpose()
    random_pos_ind = np.random.choice(np.arange(0,grid_length**2), N, replace=False)
    random_pos = all_pos[random_pos_ind]
    
    random_pos_em = random_pos[0:N_em]
    random_pos_juv = random_pos[N_em:(N_em + N_juv)]
    random_pos_ad = random_pos[(N_em + N_juv):N]
    
    idx = 0
    
    for i in range(N_em):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[i])
        em_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_em[i, :], dtype=np.int64), 0)
        init_DEB(em_ind, 1, 0)
        em_ind.ingested = 0
        
        init_population[i] = em_ind
        put_individual_in_grid(em_ind, init_matrix)
        idx += 1
    
    for j in range(N_juv):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_em + j])
        juv_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_juv[j, :], dtype=np.int64), 0)
        init_DEB(juv_ind, 2, 0)
        juv_ind.ingested = 0
        
        init_population[int(N_em+j)] = juv_ind
        put_individual_in_grid(juv_ind, init_matrix)
        idx += 1
    
    for k in range(N_ad):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_em + N_juv + k])
        ad_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_ad[k, :], dtype=np.int64), 0)
        init_DEB(ad_ind, 3, 0)
        ad_ind.ingested = 0
        
        init_population[int(N_em+N_juv+k)] = ad_ind
        put_individual_in_grid(ad_ind, init_matrix)
        idx += 1
    
    max_id = idx - 1
    init_df = pop2df(init_population, df, 0)
    return init_matrix, init_population, init_df, max_id

def constructInitMatrixDaphniaCase(init_individuals, grid_length, DEB_p, gen_init_params, behave_p, init_temperature, init_food_density, seed=None, max_id=0):
    '''
    Function that constructs the initial matrix for the simulation. Initializes the embryos, juveniles and adults in the grid. Embryos follow the case study where they are initialized as neonates with a random development time between 0 and 1 day.

    Parameters
    ----------
    init_individuals : list
        List of initial number of individuals in the grid. [N_em, N_juv, N_adu]
    grid_length : int
        Length of the grid.
    DEB_p : list
        List of DEB parameters.
    gen_init_params : list
        List of initial parameters for the genetics of individuals.
    behave_p : list
        List of behavioural parameters.
    init_temperature : float
        Initial temperature of the grid.
    init_food_density : float
        Initial food density of the grid.
    seed : int, optional
        Seed for random number generator. The default is None.
    max_id : int, optional
        Maximum id of individuals. The default is 0.

    Returns
    -------
    init_matrix : numpy array
        Initial matrix of the grid.
    init_population : numpy array
        Initial population of the grid.
    df : pandas dataframe
        Dataframe to store the data of the individuals.
    '''
    # define custom numpy dtype
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    
    # Initialize grid
    init_matrix = np.zeros((grid_length, grid_length), dtype=patchdt)
    init_population = np.zeros(np.sum(init_individuals), dtype=object)
    df = pd.DataFrame({'timestep': [], 'id': [], 'age': [], 'stage': [], 'L': [], 'E': [], 'R': [], 'fecundity': [], 'potential_fecundity': [], 'H': [], 'E_0': [], 'starved': [], 'generation': [], 'time_spent_handling': [], 'time_spent_searching': [], 'ingested': [], 'f_resp_5min': [], 'p_Am': [], 'x_pos': [], 'y_pos': [], 'neighbours': []})
    
    init_matrix['food'] = init_food_density
    init_matrix['temperature'] = init_temperature
    
    # Define number of initial population
    N_em = init_individuals[0]
    N_juv = init_individuals[1]
    N_ad = init_individuals[2]
    N = N_em + N_juv + N_ad
    
    #fix seed if seed given
    if seed:
        rs = np.random.RandomState(seed)
        seeds = rs.choice(999999, size=N, replace=False)
    else:
        seeds = np.random.choice(999999, size=N, replace=False)

    
    # Determining random positions in the grid & making sure they're unique
    all_pos = np.indices((grid_length, grid_length)).reshape((2,grid_length**2)).transpose()
    random_pos_ind = np.random.choice(np.arange(0,grid_length**2), N, replace=False)
    random_pos = all_pos[random_pos_ind]
    
    random_pos_em = random_pos[0:N_em]
    random_pos_juv = random_pos[N_em:(N_em + N_juv)]
    random_pos_ad = random_pos[(N_em + N_juv):N]
    
    idx = 0
    
    # Random development times between 0 and 24 h
    development_times = np.random.uniform(0, 1, size=N)
    
    for i in range(N_em):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[i])
        em_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_em[i, :], dtype=np.int64), 0)
        init_DEB(em_ind, 1, development_times[i])
        em_ind.ingested = 0
        
        init_population[i] = em_ind
        put_individual_in_grid(em_ind, init_matrix)
        idx += 1
    
    for j in range(N_juv):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_em + j])
        juv_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_juv[j, :], dtype=np.int64), 0)
        init_DEB(juv_ind, 2)
        juv_ind.ingested = 0
        
        init_population[int(N_em+j)] = juv_ind
        put_individual_in_grid(juv_ind, init_matrix)
        idx += 1
    
    for k in range(N_ad):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_em + N_juv + k])
        ad_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array(random_pos_ad[k, :], dtype=np.int64), 0)
        init_DEB(ad_ind, 3)
        ad_ind.ingested = 0
        
        init_population[int(N_em+N_juv+k)] = ad_ind
        put_individual_in_grid(ad_ind, init_matrix)
        idx += 1
    
    max_id = idx - 1
    init_df = pop2df(init_population, df, 0)
    return init_matrix, init_population, init_df, max_id


### Reading results ###
def pop2df(population, df, t):
    '''
    Helper function that converts a list of individuals to a pandas dataframe.

    Parameters
    ----------
    population : list
        List of individuals.
    df : pandas dataframe
        Dataframe to which the individuals are added.
    t : int
        Timestep of the simulation

    Returns
    -------
    df : pandas dataframe
        Dataframe with the individuals added.
    '''
    for individual in population:
        entry = {'timestep': t, 'id': individual.id, 'age': individual.age, 'stage':  individual.stage, 'L': individual.L, 'E': individual.E, 'R': individual.R, 'fecundity': individual.fecundity, 'potential_fecundity' : individual.potential_fecundity, 'H': individual.H, 'E_0': individual.init_E, 'starved': individual.starved, 'generation': individual.generation, 'time_spent_handling': individual.time_spent_handling, 'time_spent_searching': individual.time_spent_searching, 'ingested': individual.ingested, 'f_resp_5min': individual.f_resp_5min, 'p_Am': individual.p_Am*individual.pAm_p_scatter, 'x_pos': individual.current_pos[0], 'y_pos': individual.current_pos[1], 'neighbours': individual.neighbours}
        df = df.append(entry, ignore_index=True)
    return df

def countpop(grid):
    ''' 
    Helper function that counts the number of individuals in each stage of the life cycle from a grid.

    Parameters
    ----------
    grid : numpy array
        Grid with the individuals.

    Returns
    -------
    list
        List with the number of individuals in each stage of the life cycle.
    '''

    embryos = np.sum(grid['occupant'][:,:] == 1)
    juveniles = np.sum(grid['occupant'][:,:] == 2)
    adults = np.sum(grid['occupant'][:,:] == 3)
    total = juveniles + adults
    return [embryos, juveniles, adults, total]