### Imports ###
import numpy as np
import time
import sys
import gzip
import pickle
sys.path.insert(0, 'C:\\Users\\Wissam\\Documents\\DEB-eIBM\\src')

from debeibm import runObserveEveryDayAsync, constructInitMatrixDaphniaCase


if __name__ == '__main__':
    seed = None

    # DEB parameters
    pAm_g_scatter = pAm_p_scatter = 1
    p_Am = 313.169 #J cm-2 d-1
    F_m = 30.17 #volume units cm-2 d-1
    kap_X = 0.9 # (-)
    v = 0.1858 # (-)
    kap = 0.5809 # (-)
    kap_R = 0.95 # (-)
    p_M = 1200 # (J cm-3 d-1)
    p_T = 0 # (J cm-2 d-1)
    k_J = 0.2537 # (d-1)
    E_G = 4400 # (J cm-3)
    E_Hb = 0.05464 # (J)
    E_Hp = 1.09 # (J)
    T_A = 6400 # K
    T_ref = 293 # K
    h_a = 0.0002794 # d-2
    s_G = -0.3 # -
    mu_X = 525000
    mu_E = 550000
    T_L = 273
    T_H = 308
    TAL = 10000
    TAH = 30000
    del_M = 0.264
    
    # Correction of feeding searching parameter
    L_m = p_Am*kap/p_M
    Lb = 0.02578059602575623
    cell_length = (L_m + Lb)/2/del_M # setting the cell length
    F_m_cell = F_m / (cell_length**3 * 0.001) # converting volume units to cell units
    
    DEB_p = np.array([pAm_g_scatter, pAm_p_scatter, p_Am, F_m_cell, kap_X, v, kap, kap_R, p_M, p_T, k_J, E_G, E_Hb, E_Hp, T_A, T_ref, h_a, s_G, mu_X, mu_E, T_L, T_H, TAL, TAH])
    
    # Behavior parameters
    brood_size = 1
    behave_p = np.array([brood_size])
    mr = 0 # 0: random walk, 1: biased random walk, 2: maximum search, 3: zonal
    scheduling = "random" #random updating, size-priority updating
    
    # Genetic parameters
    h = 0
    iv = 0.15
    mean_pAm_g_scatter = 1
    mean_neutral_g = 1
    
    gen_init_params = np.array([h, iv, mean_pAm_g_scatter, mean_neutral_g])
    
    # IBM system parameters
    grid_length = 155
    grid_density = 0.5
    init_individuals = np.array([5, 0, 0])
    total_time = 60
    delta_t = 5/24/60 # maximum handling time of 1/2 hour (the organism can eat a particle of a size that would take this handling time (max particle size))
    
    # Feeding conversion
    mu_X = 525000
    mu_E = 550000
    y_E_X = kap_X * mu_X/mu_E #yield of reserve on food
    y_X_E = 1/y_E_X
    K = (p_Am*y_X_E)/F_m_cell # Half saturation constant in terms of energy food
    
    nX = np.array([1, 1.8, 0.5, 0.15]) # mol per c-mol
    wX = np.array([12, 1, 16, 14]) # g per mol
    W = np.sum(nX*wX) # g food per c-mol
    
    # Environment parameters
    init_temperature = 293 # K
    end_temperature = init_temperature
    times = np.arange(0, total_time, delta_t)
    temperatures = np.ones(int(total_time/delta_t)+2)*init_temperature
    
    # Food transformations
    volume_per_grid = cell_length**3*grid_length**2 * 0.001 # volume liters per grid
    volume = volume_per_grid * 1000
    
    C_concentration = 0.5 # mg C per l
    C_mass = C_concentration*volume_per_grid #mg C (in grid volume ml)
    C_mole = C_mass / 12000 # mg C / mg C per mole -> C-mole (in grid volume ml)
    food_energy_grid = mu_X * C_mole # joule per c-mol x c-mol (in grid volume ml)
    food_per_cell = food_energy_grid / (grid_length**2) # joule (in grid volume ml) / (cells in grid volume)
    
    food_diffusion_rate = 0.005 #cm² per day
    start = time.time()
    init_matrix, init_population, init_df, max_id = constructInitMatrixDaphniaCase(init_individuals, grid_length, DEB_p, gen_init_params, behave_p, init_temperature, food_per_cell, seed)
    end = time.time()
    print("time to complete initial construction: " + str((end-start)/60))
    
    start = time.time()
    grids, popu, df, ts = runObserveEveryDayAsync(init_matrix, init_population, init_df, total_time, delta_t, temperatures, food_per_cell, food_diffusion_rate, h, iv, cell_length, max_id, mr, scheduling, with_tqdm=True)
    end = time.time()
    print("time to complete simulation: " + str((end-start)/60))

    # Saving the data
    filename = "../data/debeibm_grids.npy.gz"
    f = gzip.GzipFile(filename, "w")
    np.save(f, grids)
    f.close()
    
    filename = "../data/debeibm_df.pickle"
    with open(filename, 'wb') as f:
        pickle.dump(df, f)