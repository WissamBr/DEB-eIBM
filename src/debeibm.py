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
def DEB_normal_em_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
def DEB_normal_em(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables of an embryo and food (y) based on its DEB parameters and temperature.
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
def DEB_normal_juv_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
    
    X_conc = X/volume
    f = (X_conc/(K+X_conc))

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
def DEB_normal_juv(t, y, DEB_p, temperature):
    '''
    Calculates the change in DEB state variables for a juvenile organism and food availability based on the DEB parameters and temperature.
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
    Calculate the change in DEB state variables for an adult organism and the food availability based on the DEB parameters and the current state variables
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
def DEB_normal_adu_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
        
    X_conc = X/volume
    f = (X_conc/(K+X_conc))

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

@jit(nopython=True)
def DEB_starved_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
        
    X_conc = X/volume
    f = (X_conc/(K+X_conc))
    
    
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
    '''
    # Define DEB parameters
    E_Hb = DEB_p[12]                           # maturity at birth (J)
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y

    
    return E_H - E_Hb

@eventAttr(True, 1)
def reach_birth_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
    '''
    # Define DEB parameters
    E_Hp = DEB_p[13]                            # maturity at birth (J)
    
    # Definition of the state variables
    L, E, E_H, E_R, q, h, X = y
    
    return E_H - E_Hp

@eventAttr(True, 1)
def reach_puberty_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
def reach_starvation_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
def reach_normal_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
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
def reach_food_empty(t, y, DEB_p, temperature):
    '''
    To be removed
    '''
    return y[6]

@eventAttr(True, -1)
def reach_food_empty_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
    '''
    return y[6]

@eventAttr(True, -1)
def reach_death(t, y, DEB_p, temperature):
    '''
    Event function that stops the ode solver when the organism reaches reserve-related death conditions, defined as E < 0.
    '''
    return y[1]

@eventAttr(True, -1)
def reach_death_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
    '''
    return y[1]

@eventAttr(False, 1)
def empty_event(t, y, DEB_p, temperature):
    '''
    Example of an event function that does not stop the ode solver.
    '''
    return 1

@eventAttr(False, 1)
def empty_event_volume(t, y, DEB_p, temperature, volume):
    '''
    To be removed
    '''
    return 1


#%% Check starvation
def check_starvation(DEB_p, y):
    '''
    Checks if the organism is in starvation conditions, defined as the moment when scaled reserve density e is below the scaled length l (e < l).
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
    Returns the time t and modifies the individual's state variables in place.
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
    Returns the time t and modifies the individual's state variables in place.
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


def solve_DEB_total_volume(individual, temperature, delta_t, food, volume, food_mode):
    '''
    To be removed
    '''

    # Determine the food mode, f for functional response, X for explicity food density
    if food_mode == "f":
        f = food
        K = (individual.p_Am*individual.pAm_p_scatter/(individual.kap_X*individual.F_m))
        
        X = f*K/(1.000000001-f)
        
    if food_mode == "X":
        X = food
        if X <= 0:
            X = 0
        
    # Define initial conditions
    t_end = delta_t
    t = 0
    y = np.copy(individual.DEB_v)
    y = np.append(y, X)
    
    # Initialize events (if it's an embryo it can reach birth and puberty, if it's a juv it can only reach puberty, if it's an adult these events are not relevant)
    if individual.stage == 1:
        reach_birth_current = reach_birth_volume
        reach_puberty_current = reach_puberty_volume
    elif individual.stage == 2:
        reach_birth_current = empty_event_volume
        reach_puberty_current = reach_puberty_volume
    elif individual.stage == 3:
        reach_birth_current = empty_event_volume
        reach_puberty_current = empty_event_volume
        
    reach_starvation_current = reach_starvation_volume
    reach_normal_current = reach_normal_volume
    
    if X > 0:
        reach_food_empty_current = reach_food_empty_volume
    elif X == 0:
        reach_food_empty_current = empty_event_volume
    
    # Initialize the correct DEB system
    if individual.starved:
        ode_f = DEB_starved_volume
        
        if individual.stage == 2:
            reach_birth_current = empty_event_volume
        elif individual.stage == 3:
            reach_puberty_current = empty_event_volume
        elif individual.stage == 1:
            ode_f = DEB_normal_em_volume
            
    elif individual.stage == 1:
        ode_f = DEB_normal_em_volume
        
    elif individual.stage == 2:
        ode_f = DEB_normal_juv_volume
        reach_birth_current = empty_event_volume
        
    elif individual.stage == 3:
        ode_f = DEB_normal_adu_volume
        reach_puberty_current = empty_event_volume
    
    #switched_current = 0
    while True:
        
        sol = solve_ivp(ode_f, [t, t_end], y, args=(individual.DEB_p, temperature, volume), events=(reach_birth_current, reach_puberty_current, reach_starvation_current, reach_normal_current, reach_death_volume, reach_food_empty_current), dense_output=True)
        
        if sol.status == 1:
            # if sol.t[0] == sol.t[-1]:
            #     print("stuck")
            # event 1: embryo to juvenile
            if len(sol.y_events[0]) > 0:
                
                # time and state at birth
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # set the right function
                ode_f = DEB_normal_juv_volume
                
                # save and update individual life-history
                individual.stage = 2
                individual.lb = y[0]
                individual.ab = individual.age + t
                
                reach_birth_current = empty_event_volume
                
            # event 2: juvenile to puberty
            elif len(sol.y_events[1]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                ode_f = DEB_normal_adu_volume
                individual.stage = 3
                individual.lp = y[0]
                individual.ap = individual.age + t
                
                reach_puberty_current = empty_event_volume
            # event 3: normal to starvation
            elif len(sol.y_events[2]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                ode_f = DEB_starved_volume
                individual.starved = True
                
                # Because of the fact that food during the simulation of DEB cannot increase, when an individual starves, it cannot go back to normal within its simulation so both events are shutdown
                reach_starvation_current = empty_event_volume
                reach_normal_current = empty_event_volume

                    
            # event 4: starvation to normal
            elif len(sol.y_events[3]) > 0:
                t = sol.t[-1]
                y = sol.y[:, -1].copy()
                
                # calculate the e_scaled and l_scaled
                if individual.stage == 2:
                    ode_f = DEB_normal_juv_volume
                elif individual.stage == 3:
                    ode_f = DEB_normal_adu_volume
                individual.starved = False
                
                # An individual can go from starvation to normal then back to starvation in one simulation, but it can't go back to normal
                reach_normal_current = empty_event_volume
                reach_starvation_current = reach_starvation_volume

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
                reach_food_empty_current = empty_event_volume
        else:
            t = t_end
            break
    
    # store results
    individual.DEB_v = sol.y[:6, -1]
    individual.ingested += food - sol.y[6,-1]
    ingested_by_individual = food - sol.y[6,-1]

    # if at the end, reserve is still negative, make em dead
    if individual.E < 0:
        #print("Starvation Death")
        individual.dead = True
    
    # age up
    individual.age += t
        
    return t, ingested_by_individual

#%%
def init_DEB_volume(individual, stage, development_time):
    # determine initial reserve
    # simulate until the stage they reach == input stage (j,a)
    # get state variables at that point

    '''
    To be removed
    '''

    # Determine initial reserve
    E0 = init_reserve_DEB(individual.DEB_p, 1)
    individual.init_E = E0
    
    # If stage == embryo, then just standard initial conditions
    if  stage == 1:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0], dtype=np.float64))
        individual.age = 0
        
        while individual.age < development_time:
            _ = solve_DEB_total_volume(individual, temperature = 293, delta_t = 1/24, food = 1, volume=1, food_mode = "f")
            
    # If stage == juvenile, simulate the DEB from embryo to juvenile
    elif stage == 2:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0]))
        individual.age = 0

        while individual.stage < 2:
            _ = solve_DEB_total_volume(individual, temperature = 293, delta_t = 1/24, food = 1, volume=1, food_mode = "f")
            if individual.dead:
                raise Exception("Individual never reaches juvenile stage!")
                
        
        juv_age = individual.age
        
        while individual.age < (juv_age + development_time):
            _ = solve_DEB_total_volume(individual, temperature = 293, delta_t = 1/24, food = 1, volume=1, food_mode = "f")

    # If stage == adult, simulate the DEB from embryo to adult
    elif stage == 3:
        individual.DEB_v = (np.array([10**-5 , E0, 0, 0, 0, 0], dtype=np.float64))
        individual.age = 0

        while individual.stage < 3:
            _ = solve_DEB_total_volume(individual, temperature = 293, delta_t = 1/24, food = 1, volume=1, food_mode = "f")

            if individual.dead:
                raise Exception("Individual never reaches adult stage!")
        
        adu_age = individual.age
        
        while individual.age < (adu_age + development_time):
            _ = solve_DEB_total_volume(individual, temperature = 293, delta_t = 1/24, food = 1, volume=1, food_mode = "f")

def init_DEB(individual, stage, development_time):
    # determine initial reserve
    # simulate until the stage they reach == input stage (j,a)
    # get state variables at that point
    '''
    stage: 1 = embryo, 2 = juvenile, 3 = adult
    individual: individual object
    development_time: time to simulate

    Returns: individual object with DEB_v and age updated

    Initializes the DEB system of an individual by simulating it until it reaches the stage specified by the input stage and development time (if the chosen stage is embryo)
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
    herit: heritability of the trait (0-1)
    cv: coefficient of variation of the trait
    mean_pAm_g_scatter: mean of the trait in the population
    mean_neutral_g: mean of the neutral trait in the population
    seed: seed for the random number generator

    Returns: mean_pAm_g_scatter_ln, mean_neutral_g_ln, mean_env_ln, g_var, e_var

    Initializes the traits of the population by transforming the lognormally distributed traits to normal distribution parameters and calculating the genetic and environmental variance
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
def init_behavior(brood_size):
    '''
    To be removed
    '''
    behave_p = np.array([brood_size])
    return behave_p

#%%
def checkDeath(individual, delta_t):
    '''
    Checks if the individual dies due to starvation or age
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
    individual: individual object
    grid_new: 2D grid

    Returns the empty cells in the (Moore) neighbourhood of the individual
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
def determineZoneOfInfluence(individual, grid_new, population, delta_t):
    '''
    To be removed
    '''

    # Determine the radius
    temperature = grid_new["temperature"][int(individual.current_pos[1]), int(individual.current_pos[0])]
    gamma_t = (1 + math.exp((individual.T_AL/temperature) - (individual.T_AL/individual.T_L)) + math.exp((individual.T_AH/individual.T_H) - (individual.T_AH/temperature)))
    gamma_t_ref = (1 + math.exp((individual.T_AL/individual.T_ref) - (individual.T_AL/individual.T_L)) + math.exp((individual.T_AH/individual.T_H) - (individual.T_AH/individual.T_ref)))

    arrhenius_corr = math.exp((individual.T_A/individual.T_ref)-(individual.T_A/temperature))*gamma_t_ref/gamma_t 
    F_m_corr = individual.F_m*arrhenius_corr
    
    speed = F_m_corr*individual.L**2
    cell_radius = math.floor(speed * delta_t)
    
    # Define grid window of that zone by padding original grid
    grid_length = grid_new.shape[1]
    positions = np.array([(int(individual.current_pos[1]) - cell_radius), grid_length - (int(individual.current_pos[1]) + cell_radius),(int(individual.current_pos[0]) - cell_radius),grid_length - (int(individual.current_pos[0]) + cell_radius)])
    positions_padded = positions + cell_radius
    
    # Output list of individuals in zone
    local_occupants = [ind for ind in population if (ind.current_pos[1] + cell_radius >= positions_padded[0]) & (ind.current_pos[1] + cell_radius < positions_padded[1]) & (ind.current_pos[0] + cell_radius >= positions_padded[2]) & (ind.current_pos[0] + cell_radius < positions_padded[3])]
    
    return local_occupants

#%%
def interferenceTaxisWeights(individual, grid_new):
    '''
    To be removed
    '''
    # Get radius
    
    # Get individuals in radius that are bigger than focal individual
    
    # determine distance in cells to that individual
    
    # output ratio of distance to individual over radius
    
    # set weights in cells opposite to individual as uniform, remaining zero
    
    pass

#%% 
def determineOffspringParams(individual, herit, iv):
    '''
    Determines the offspring parameters based on the parent parameters and the heritability (introducing variation)
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
    Determines the reproduction threshold based on the offspring parameters and the heritability
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
    Moves the individual on the grid to a random empty spot in its neighbourhood (random walk)
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

def reproduceNonSpatial(individual, population, max_id):
    ''' Reproduces an offspring on the grid on a random empty spot in its neighbourhood '''
    # reproduce as many offspring as your brood_size
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
    pos = individual.current_pos
    
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
    
    # remove egg cost from buffer
    individual.R -= individual.offspring_bank[0, 4]
    
    # add to the fecundity counter of the individual
    individual.fecundity += 1
     
    return population, max_id

def reproduce(grid, individual, population, empty_neighbourhood, max_id):
    ''' Reproduces an offspring on the grid on a random empty spot in its neighbourhood '''
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

@jit(nopython=True)
def updateEnvironmentEveryStepConstantRenewal(grid, food_capacity, time_to_regen, food_diffusion_rate, temperature, delta_t, cell_length, t):    
    
    if round(t, 4) % time_to_regen == 0:
        grid['food'][:, :] = food_capacity
    
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
    t_in = np.arange(0, 1, dt)
    
    # if the IBM step is smaller than the stability criterium for stable timesteps, pick the IBM timestep
    if delta_t < dt:
        dt = delta_t
    
    # Diffusion
    for m in range(len(t_in)):
        u_ext[1:-1, 1:-1] = u_ext0[1:-1, 1:-1] + D * dt * ((u_ext0[2:, 1:-1] - 2*u_ext0[1:-1, 1:-1] + u_ext0[:-2, 1:-1])/dx2 + (u_ext0[1:-1, 2:] - 2*u_ext0[1:-1, 1:-1] + u_ext0[1:-1, :-2])/dy2 )
        u_ext0 = np.copy(u_ext)
    
    grid['food'][:, :] = np.copy(u_ext[1:-1, 1:-1])
    
@jit(nopython=True)
def updateFoodStockDailyAddition(food_stock_new, food_capacity, t):    
    if round(t, 4) % 1 == 0:
        food_stock_new += food_capacity

#%% Forage functions

def forageAsync(individual, grid_new, delta_t, mr):
    
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

def forageFoodSync(individual, grid_new, delta_t):
    
    # Define some key parameters
    y_E_X = individual.kap_X * individual.mu_X/individual.mu_E # yield of reserve on food
    y_X_E = 1/y_E_X # yield of food on reserve
    
    # initialize time spent foraging
    time_spent_foraging = 0
    total_ingestion = 0
    
    # temporary food grid
    grid_tmp = np.copy(grid_new)
    
    while time_spent_foraging < delta_t:
        # as long the individual is not in a cell with food it will move around till it finds food or can't move
        distance_covered = 0
        time_spent_searching_current_cycle = 0
        while grid_tmp["food"][individual.current_pos[0], individual.current_pos[1]] == 0:
            # determine empty neighbourhood
            empty_neighbourhood = determineNeighbourhood(individual, grid_tmp['occupant'][:, :])
            
            # if empty neighbourhood is null then proceed to the handling
            if len(empty_neighbourhood) == 0:
                break
            else:
                # otherwise move
                move(grid_tmp, individual, empty_neighbourhood, biased = True)
                distance_covered += 1
        
                # calculate time spent on searching/moving
                speed = individual.F_m * individual.L**2 # cells per day
                time_spent_searching = 1/speed #cells divided by cells per day = days
                time_spent_foraging += time_spent_searching
                time_spent_searching_current_cycle += time_spent_searching
                
                # if time spent on foraging reaches delta_t during the searching then stop searching and move on
                if time_spent_foraging >= delta_t:
                    time_spent_foraging = delta_t
                    break
        
        # if time spent on foraging after searching takes up all the available time in delta t, move on and simulate deb
        if time_spent_foraging >= delta_t:
            # break out of the while loop
            break
        
        # how much does it instantaneously eat over a remaining handling period delta_t-time_spent_foraging
        max_ingestion = individual.p_Am*y_X_E*individual.pAm_p_scatter*(delta_t-time_spent_foraging)*individual.L**2 #Max ingestion in J food
        food = grid_tmp["food"][individual.current_pos[0], individual.current_pos[1]]
        
        # add maximum ingestion rate if possible otherwise add whatever is there
        if (food - max_ingestion <= 0):
            ingestion = food
        else:
            ingestion = max_ingestion
        
        individual.E += ingestion*y_E_X
        food_updated = food - ingestion
        total_ingestion += ingestion
        
        # determine whether the organism is starving or not
        if check_starvation(individual.DEB_p, individual.DEB_v):
            individual.starved = True
        else:
            individual.starved = False
        
        # time spent handling food
        time_spent_handling = ingestion*y_E_X/(individual.p_Am*individual.pAm_p_scatter*individual.L**2)
        
        # if the neighbourhood is filled then the individual will spend all its time handling food
        empty_neighbourhood = determineNeighbourhood(individual, grid_tmp['occupant'][:, :])
        if len(empty_neighbourhood) == 0:
            time_spent_handling = delta_t - time_spent_foraging
        
        # # forward time with time spent handling and searching of current cycle
        # step_size = time_spent_handling + time_spent_searching_current_cycle
        # temperature = grid_new["temperature"][individual.current_pos[0], individual.current_pos[1]]
        # _ = solve_DEB_catabolism(individual, temperature, step_size, 1, food_mode = "X")
        
        # update grid and total time spent foraging
        #put_individual_in_grid(individual, grid_new)
        grid_tmp["food"][individual.current_pos[0], individual.current_pos[1]] = food_updated
        time_spent_foraging += time_spent_handling
    
    #forward DEB state variables in time
    temperature = grid_tmp["temperature"][individual.current_pos[0], individual.current_pos[1]]
    _ = solve_DEB_catabolism(individual, temperature, time_spent_foraging, 1, food_mode = "X")
    individual.time_spent_searching = time_spent_searching_current_cycle
    individual.time_spent_handling = time_spent_foraging - time_spent_searching_current_cycle
    individual.ingested = total_ingestion
    put_individual_in_grid(individual, grid_new)
        
    return grid_new

    
#%% UPDATE FUNCTIONS

def updateNonSpatialSync(food_stock, population, t, delta_t, temperature, food_capacity, herit, iv, max_id, volume):
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
    # New Update
    food_stock_new = np.copy(food_stock)
    
    # Update environment
    updateFoodStockDailyAddition(food_stock_new, food_capacity, t)
    
    # Scheduling (make function out of this)
    np.random.shuffle(population)
    #population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Taking random individuals random.choice (but without replacement)!
    # Forwarding the individuals in the grid in time
    index = 0
    total_ingested = 0
    for individual in population:
                
        # update energy budget and determine absolute amount of food ingested based on food stock (concentration is calculated using volume in DEB)
        _, ingested_by_individual = solve_DEB_total_volume(individual, temperature, delta_t, food_stock_new, volume, food_mode = "X")
        total_ingested += ingested_by_individual
        
        # Check for individual death (starvation & ageing)
        checkDeath(individual, delta_t)
        if individual.dead:
            population = np.delete(population, index) # remove from individual from the population list
            continue
        
        # Inheritance and offspring params if adult and empty bank or impossible parameterset (rep threshold is zero)
        if (individual.stage == 3) & (individual.rep_threshold == 0):
            determineOffspringParams(individual, herit, iv)
            
            
        # Determine reproduction threshold if adult and offspring params ready in bank (rep threshold is -1)
        if (individual.stage == 3) & (individual.rep_threshold < 0):
            determineRepThreshold(individual, herit, iv)
            
        
        # reproduce if enough buffer
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold != 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduceNonSpatial(individual, population, max_id)
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
        
        # Move forward the index of the for loop
        index += 1
    
    # Update feeding (synchronous)
    food_stock_new -= total_ingested
    
    # Forward outer time with delta_t
    t += delta_t
    
    return [food_stock_new, population, max_id]

def individualForaging(individual, grid_new, delta_t):
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
    # Check for individual death (starvation & ageing)
    checkDeath(individual, delta_t)
    if individual.dead:
        remove_individual_from_grid(individual, grid_new) # remove individual data from the grid
        population = np.delete(population, index) # remove from individual from the population list
    
    return individual, grid_new, population
    

def IBMschedule(population, grid_new, delta_t, herit, iv, max_id, mr):
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
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
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
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
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

def updateAsyncSizePriority(grid, population, t, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
    # New Update
    grid_new = np.copy(grid)
    
    # Update environment
    # All cells get food added equal to the f1_equivalent when env_update_timer hits 7 (days)
    updateEnvironmentDailyConstantAddition(grid_new, food_capacity, food_diffusion_rate, temperature, delta_t, cell_length, t)
    
    # Scheduling (make function out of this)
    #np.random.shuffle(population)
    population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Taking random individuals random.choice (but without replacement)!
    # Forwarding the individuals in the grid in time
    index = 0
    for individual in population:
        
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
        
        # Determine reproduction threshold if necessary
        if (individual.rep_threshold == 0) & (individual.stage == 3):
            determineRepThreshold(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
        
        # reproduce if enough buffer
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold != 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduce(grid_new, individual, population, empty_neighbourhood, max_id)
                empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
            continue
    
    # Forward outer time with delta_t
    t += delta_t
    
    return [grid_new, population, max_id]

### Update function of IBM ###
def updateFoodSync(grid, population, t, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
    # New Update
    grid_new = np.copy(grid)
    
    # Update environment
    # All cells get food added equal to the f1_equivalent when env_update_timer hits 7 (days)
    updateEnvironmentDailyConstantAddition(grid_new, food_capacity, food_diffusion_rate, temperature, delta_t, cell_length, t)
    
    # Scheduling (make function out of this)
    np.random.shuffle(population)
    #population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Taking random individuals random.choice (but without replacement)!
    # Forwarding the individuals in the grid in time
    index = 0
    for individual in population:
        
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
                grid_new = forageFoodSync(individual, grid_new, (delta_t - t_forage))
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
        
        # Determine reproduction threshold if necessary
        if (individual.rep_threshold == 0) & (individual.stage == 3):
            determineRepThreshold(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
        
        # reproduce if enough buffer
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold != 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduce(grid_new, individual, population, empty_neighbourhood, max_id)
                empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
            continue
    
    # Forward outer time with delta_t
    t += delta_t
    
    return [grid_new, population, max_id]

### Update function of IBM ###
def updateAsyncConstantRenewal(grid, population, t, delta_t, temperature, food_capacity, time_to_regen, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
    # New Update
    grid_new = np.copy(grid)
    
    # Update environment
    # All cells get food added equal to the f1_equivalent when env_update_timer hits 7 (days)
    updateEnvironmentEveryStepConstantRenewal(grid_new, food_capacity, time_to_regen, food_diffusion_rate, temperature, delta_t, cell_length, t)
    
    # Scheduling (make function out of this)
    np.random.shuffle(population)
    #population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Taking random individuals random.choice (but without replacement)!
    # Forwarding the individuals in the grid in time
    index = 0
    for individual in population:
        
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
        
        # Determine reproduction threshold if necessary
        if (individual.rep_threshold == 0) & (individual.stage == 3):
            determineRepThreshold(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
        
        # reproduce if enough buffer
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold != 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduce(grid_new, individual, population, empty_neighbourhood, max_id)
                empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
            continue
    
    # Forward outer time with delta_t
    t += delta_t
    
    return [grid_new, population, max_id]

### Update function of IBM ###
def updateFoodSyncConstantRenewal(grid, population, t, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Updates the grid and individuals matrices'''
    '''Returns the updated grid and individual matrix'''
        
    # New Update
    grid_new = np.copy(grid)
    
    # Update environment
    # All cells get food added equal to the f1_equivalent when env_update_timer hits 7 (days)
    updateEnvironmentEveryStepConstantRenewal(grid_new, food_capacity, food_diffusion_rate, temperature, delta_t, cell_length, t)
    
    # Scheduling (make function out of this)
    np.random.shuffle(population)
    #population = sorted(list(population), key=lambda x: x.L, reverse=True) # size priority scheduling
    
    # Taking random individuals random.choice (but without replacement)!
    # Forwarding the individuals in the grid in time
    index = 0
    for individual in population:
        
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
                grid_new = forageFoodSync(individual, grid_new, (delta_t - t_forage))
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
        
        # Determine reproduction threshold if necessary
        if (individual.rep_threshold == 0) & (individual.stage == 3):
            determineRepThreshold(individual, herit, iv)
            put_individual_in_grid(individual, grid_new)
        
        # reproduce if enough buffer
        if (individual.stage == 3) & (individual.R > individual.rep_threshold) & (individual.rep_threshold != 0):
            
            # reproduce to empty out entire buffer
            while individual.R > individual.rep_threshold:
                population, max_id = reproduce(grid_new, individual, population, empty_neighbourhood, max_id)
                empty_neighbourhood = determineNeighbourhood(individual, grid_new['occupant'][:, :])
           
            # clear out the bank after reproduction occurred
            individual.offspring_bank[0, :] = np.zeros((5), dtype=np.float64)
            
            # Reset rep_threshold
            individual.rep_threshold = 0
            continue
    
    # Forward outer time with delta_t
    t += delta_t
    
    return [grid_new, population, max_id]


#%% RUN FUNCTIONS

### Run function of IBM ###
def runObserveEveryStepAsync(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((int(round(total_time/delta_t)) + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([int(round(total_time/delta_t)) + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    for t in (ts[1:]):
        
        # observe food and population size before update
        grids[index] = np.copy(grid_current)
        N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
        N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
        N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
        popu[index] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateAsync(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr)
        
        # observe individual response after update
        df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

### Run function of IBM ###
def runObserveEveryStepFoodSync(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((int(round(total_time/delta_t)) + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([int(round(total_time/delta_t)) + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    for t in (ts[1:]):
        
        # observe food and population size before update
        grids[index] = np.copy(grid_current)
        N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
        N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
        N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
        popu[index] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateFoodSync(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id)
        
        # observe individual response after update
        df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def runObserveEveryStepAsyncConstantRenewal(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((int(round(total_time/delta_t)) + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([int(round(total_time/delta_t)) + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    for t in (ts[1:]):
        
        # observe food and population size before update
        grids[index] = np.copy(grid_current)
        N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
        N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
        N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
        popu[index] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateAsyncConstantRenewal(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id)
        
        # observe individual response after update
        df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

### Run function of IBM ###
def runObserveEveryStepFoodSyncConstantRenewal(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    grid_current = np.copy(init_matrix)
    population_current = np.copy(init_population)
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # we consider 11 spatially explicit outputs (individual, food, temperature)
    grids = grid_current.reshape(1, init_matrix.shape[0], init_matrix.shape[1])
    patchdt = np.dtype([('occupant', np.int64), ('food', np.float64), ('temperature', np.float64)])
    grids = np.zeros((int(round(total_time/delta_t)) + 1,init_matrix.shape[0], init_matrix.shape[1]), dtype= patchdt)
    grids[0] = init_matrix
    index = 1
    
    # initialization of dataframe
    popu = np.zeros([int(round(total_time/delta_t)) + 1, 4], dtype=int)
    popu[0, :] = countpop(init_matrix)
    
    for t in (ts[1:]):
        
        # observe food and population size before update
        grids[index] = np.copy(grid_current)
        N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
        N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
        N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
        popu[index] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateFoodSyncConstantRenewal(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id)
        
        # observe individual response after update
        df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def resetFecundityIngestion(population):
    for individual in population:
        individual.potential_fecundity = 0
        individual.fecundity = 0
        individual.ingested = 0
        individual.neighbours = 0

def runObserveEveryDayAsync(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling, with_tqdm=False):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
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

def runObserveEveryDayAsyncConstantRenewal(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, time_to_regen, food_diffusion_rate, herit, iv, cell_length, max_id, with_tqdm = False):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
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
            grid_new, population_new, max_id = updateAsyncConstantRenewal(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, time_to_regen, food_diffusion_rate, herit, iv, cell_length, max_id)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
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
            grid_new, population_new, max_id = updateAsyncConstantRenewal(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, time_to_regen, food_diffusion_rate, herit, iv, cell_length, max_id)
            
            # observe individual response after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                
            index += 1
            
            # replace current grid with new grid
            grid_current = np.copy(grid_new)
            population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def runObserveEveryDayAsyncLogisticGrowth(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_growth_rate, food_diffusion_rate, herit, iv, cell_length, max_id, mr, scheduling, with_tqdm=False):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
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

def runObserveEveryDayAsyncSizePriority(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
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
    
    for t in (ts[1:]):
        
        # observe food and population size at start of update
        if round(t, 4) % 1 == 0:
            grids[int(round(t))] = np.copy(grid_current)
            N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
            N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
            N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
            popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateAsyncSizePriority(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id)
        
        # observe individual response after update
        if round(t, 4) % 1 == 0:
            df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def runObserveEveryDayFoodSync(init_matrix, init_population, df, total_time, delta_t, temperature, food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
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
    
    for t in (ts[1:]):
        
        # observe food and population size at start of update
        if round(t, 4) % 1 == 0:
            grids[int(round(t))] = np.copy(grid_current)
            N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
            N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
            N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
            popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        
        # update grid
        grid_new, population_new, max_id = updateFoodSync(grid_current, population_current, t, delta_t, temperature[index-1], food_capacity, food_diffusion_rate, herit, iv, cell_length, max_id)
        
        # observe individual response after update
        if round(t, 4) % 1 == 0:
            df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        grid_current = np.copy(grid_new)
        population_current = np.copy(population_new)
    
    return [grids, popu, df, ts]

def runObserveEveryStepNonSpatialSync(food_stock, init_population, df, total_time, delta_t, temperature, herit, iv, max_id, volume):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    t = 0
    food_stock_current = food_stock
    population_current = np.copy(init_population)
    
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # initialization of dataframe
    popu = np.zeros([int(round(total_time/delta_t))+1, 4], dtype=int)
    foodu = np.zeros(int(round(total_time/delta_t))+1, dtype=np.float64)
    environmentu = np.zeros((int(round(total_time/delta_t))+1, 5), dtype=np.float64)
    
    popu[0, :] = countpop_p(population_current)
    foodu[0] = food_stock_current
    environmentu[0,0] = foodu[0]
    environmentu[0,1:] = popu[0,:]
    
    index = 1
    for t in (ts[1:]):
        
        N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
        N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
        N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
        popu[index] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
        foodu[index] = food_stock_current
        environmentu[index,0] = foodu[index]
        environmentu[index,1:] = popu[index]
        
        # update grid
        food_stock_new, population_new, max_id = updateNonSpatialSync(food_stock_current, population_current, t, delta_t, temperature, food_stock, herit, iv, max_id, volume)
        
        # observe after update
        df = pop2df(population_new, df, t)
            
        index += 1
        
        # replace current grid with new grid
        food_stock_current = food_stock_new
        population_current = np.copy(population_new)
    
    return [df, environmentu, ts]

def runObserveDayNonSpatialSync(food_stock, init_population, df, total_time, delta_t, temperature, herit, iv, max_id, volume, with_tqdm):
    '''Runs the IBM and updates the grid for the amount of generations specified '''
    # initialize time & grids
    t = 0
    food_stock_current = food_stock
    population_current = np.copy(init_population)
    
    ts = np.arange(0,total_time+delta_t,delta_t)
    
    # initialization of dataframe
    popu = np.zeros([total_time + 1, 4], dtype=int)
    foodu = np.zeros(total_time + 1, dtype=np.float64)
    environmentu = np.zeros((total_time + 1, 5), dtype=np.float64)
    
    popu[0, :] = countpop_p(population_current)
    foodu[0] = food_stock_current
    environmentu[0,0] = foodu[0]
    environmentu[0,1:] = popu[0,:]
    
    index = 1
    if with_tqdm:
        for t in tqdm(ts[1:]):
            
            if round(t, 4) % 1 == 0:
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
                foodu[int(round(t))] = food_stock_current
                environmentu[int(round(t)),0] = foodu[int(round(t))]
                environmentu[int(round(t)),1:] = popu[int(round(t))]
            
            # update grid
            food_stock_new, population_new, max_id = updateNonSpatialSync(food_stock_current, population_current, t, delta_t, temperature, food_stock, herit, iv, max_id, volume)
            
            # observe after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                resetFecundityIngestion(population_new)
                
                
            index += 1
            
            # replace current grid with new grid
            food_stock_current = food_stock_new
            population_current = np.copy(population_new)
    else:
        for t in (ts[1:]):
            
            if round(t, 4) % 1 == 0:
                N_em = len(np.array([individual for individual in population_current if individual.stage == 1]))
                N_juv = len(np.array([individual for individual in population_current if individual.stage == 2]))
                N_adu = len(np.array([individual for individual in population_current if individual.stage == 3]))
                popu[int(round(t))] = np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)
                foodu[int(round(t))] = food_stock_current
                environmentu[int(round(t)),0] = foodu[int(round(t))]
                environmentu[int(round(t)),1:] = popu[int(round(t))]
            
            # update grid
            food_stock_new, population_new, max_id = updateNonSpatialSync(food_stock_current, population_current, t, delta_t, temperature, food_stock, herit, iv, max_id, volume)
            
            # observe after update
            if round(t, 4) % 1 == 0:
                df = pop2df(population_new, df, t)
                resetFecundityIngestion(population_new)
                
            index += 1
            
            # replace current grid with new grid
            food_stock_current = food_stock_new
            population_current = np.copy(population_new)
    
    return [df, environmentu, ts]


### Function that removes an individual from output grid
def remove_individual_from_grid(individual, grid):
    
    # general organism state
    grid['occupant'][individual.current_pos[0], individual.current_pos[1]] = 0
    
### Function that puts an individual in output grid
def put_individual_in_grid(individual, grid):
    
    # general organism state
    grid['occupant'][individual.current_pos[0], individual.current_pos[1]] = individual.stage
    
### Construction of initial matrix ###

def construct_init_matrix(init_individuals, grid_length, DEB_p, gen_init_params, behave_p, init_temperature, init_food_density, seed=None, max_id=0):
    
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

def constructInitMatrixNonSpatial(init_individuals, DEB_p, gen_init_params, behave_p, max_id=0, seed=None):
    

    # Initialize grid and dataset
    init_population = np.zeros(np.sum(init_individuals), dtype=object)
    df = pd.DataFrame({'timestep': [], 'id': [], 'age': [], 'stage': [], 'L': [], 'E': [], 'R': [], 'fecundity': [], 'potential_fecundity': [], 'H': [], 'E_0': [], 'starved': [], 'generation': [], 'time_spent_handling': [], 'time_spent_searching': [], 'ingested': [], 'p_Am': [], 'neighbours': []})
    
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
    
    idx = 0
    for i in range(N_em):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[i])
        em_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array([0, 0], dtype=np.int64), 0)
        init_DEB_volume(em_ind, 1, 0)
        em_ind.ingested = 0
        
        init_population[i] = em_ind
        idx += 1
    
    for j in range(N_juv):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_em + j])
        juv_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array([0, 0], dtype=np.int64), 0)
        init_DEB_volume(juv_ind, 2, 0)
        juv_ind.ingested = 0
        
        init_population[int(N_em+j)] = juv_ind
        idx += 1
    
    for k in range(N_ad):
        gen_p = init_genetics(gen_init_params[0], gen_init_params[1], gen_init_params[2], gen_init_params[3], seeds[N_juv + k])
        ad_ind = Individual(idx, DEB_p, behave_p, gen_p, 1, np.array([0, 0], dtype=np.int64), 0)
        init_DEB_volume(ad_ind, 3, 0)
        ad_ind.ingested = 0
        
        init_population[int(N_em+N_juv+k)] = ad_ind
        idx += 1
    
    max_id = idx - 1
    
    init_df = pop2df(init_population, df, 0)
    return init_population, init_df, max_id

### Reading results ###
def pop2df(population, df, t):
    for individual in population:
        entry = {'timestep': t, 'id': individual.id, 'age': individual.age, 'stage':  individual.stage, 'L': individual.L, 'E': individual.E, 'R': individual.R, 'fecundity': individual.fecundity, 'potential_fecundity' : individual.potential_fecundity, 'H': individual.H, 'E_0': individual.init_E, 'starved': individual.starved, 'generation': individual.generation, 'time_spent_handling': individual.time_spent_handling, 'time_spent_searching': individual.time_spent_searching, 'ingested': individual.ingested, 'f_resp_5min': individual.f_resp_5min, 'p_Am': individual.p_Am*individual.pAm_p_scatter, 'x_pos': individual.current_pos[0], 'y_pos': individual.current_pos[1], 'neighbours': individual.neighbours}
        df = df.append(entry, ignore_index=True)
    return df

def countpop_p(population):
    N_em = len(np.array([individual for individual in population if individual.stage == 1]))
    N_juv = len(np.array([individual for individual in population if individual.stage == 2]))
    N_adu = len(np.array([individual for individual in population if individual.stage == 3]))
    return np.array([N_em, N_juv, N_adu, N_juv + N_adu], dtype=int)

def countpop(grid):
    ''' Census the population of individuals: 0 -> embryo, 1 -> juvenile, 2 -> adult'''
    ''' Returns the 1x1x4 array with amount of individuals in each stage'''
    embryos = np.sum(grid['occupant'][:,:] == 1)
    juveniles = np.sum(grid['occupant'][:,:] == 2)
    adults = np.sum(grid['occupant'][:,:] == 3)
    total = juveniles + adults
    return [embryos, juveniles, adults, total]