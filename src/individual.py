### Imports ###
import numpy as np
from numba import types    # import the types
from numba.experimental import jitclass


# Overloading np.array to accept optionalTyping (float or None)
spec = [
    ('id', types.int64),
    ('p_Am', types.float64),
    ('F_m', types.float64),
    ('kap_X', types.float64),
    ('v', types.float64),
    ('kap', types.float64),
    ('kap_R', types.float64),
    ('p_M', types.float64),
    ('p_T', types.float64),
    ('k_J', types.float64),
    ('E_G', types.float64),
    ('E_Hb', types.float64),
    ('E_Hp', types.float64),
    ('T_A', types.float64),
    ('T_ref', types.float64),
    ('h_a', types.float64),
    ('s_G', types.float64),
    ('L', types.float64),
    ('E', types.float64),
    ('H', types.float64),
    ('R', types.float64),
    ('q', types.float64),
    ('h', types.float64),
    ('stage', types.int64),
    ('age', types.float64),
    ('rep_threshold', types.float64),
    ('dead', types.boolean),
    ('starved', types.boolean),
    ('start_pos', types.int64[:]),
    ('current_pos', types.int64[:]),
    ('movement', types.float64),
    ('brood_size', types.float64),
    ('offspring_bank', types.float64[:,:]),
    ('generation', types.int64),
    ('neighbours', types.int64),
    ('pAm_g_scatter', types.float64),
    ('pAm_p_scatter', types.float64),
    ('neutral_g', types.float64),
    ('neutral_p', types.float64),
    ('fecundity', types.int64),
    ('potential_fecundity', types.int64),
    ('ingested', types.float64),
    ('init_E', types.float64),
    ('lb', types.float64),
    ('lp', types.float64),
    ('lm', types.float64),
    ('ab', types.float64),
    ('ap', types.float64),
    ('am', types.float64),
    ('time_spent_searching', types.float64),
    ('time_spent_handling', types.float64),
    ('ingested', types.float64),
    ('mu_X', types.float64),
    ('mu_E', types.float64),
    ('T_L', types.float64),
    ('T_H', types.float64),
    ('T_AL', types.float64),
    ('T_AH', types.float64),
    ('f_resp_5min', types.float64),
]

### Individual Class ###
@jitclass(spec)
class Individual():
    def __init__(self, idx, DEB_p, behave_p, gen_p, stage, pos, generation):
        # DEB_params are the DEB parameters
        # behave_params are the parameters that relate to behaviour (e.g. brood size and movement rate)
        # gen_params are the parameters that relate to genetic properties (e.g. genetic variability of traits)
        
        # organism state #
        self.id = idx
        self.stage = stage
        self.age = 0
        self.rep_threshold = 0
        self.dead = False
        self.starved = False

        ## IBM-related attributes ##
        self.start_pos = pos
        self.current_pos = pos
        self.brood_size = behave_p[0]
        self.offspring_bank = np.zeros((int(self.brood_size), 5), dtype=np.float64)
        self.generation = generation
        self.fecundity = 0
        self.potential_fecundity = 0
        self.movement = 0.0
        self.neighbours = 0
        self.ingested = 0
        self.lb = -1
        self.lp = -1
        self.lm = -1
        self.ab = -1
        self.ap = -1
        self.am = -1
        self.time_spent_searching = 0
        self.time_spent_handling = 0
        self.ingested = 0
        self.f_resp_5min = 0
        self.init_E = 0
        
        ## genetics related parameters ##
        self.pAm_g_scatter = gen_p[0]
        self.pAm_p_scatter = gen_p[1]
        self.neutral_g = gen_p[2]
        self.neutral_p = gen_p[3]
        
        ## DEB-related attributes ##
        # DEB parameters #
        self.p_Am = DEB_p[2]
        self.F_m = DEB_p[3]
        self.kap_X = DEB_p[4]
        self.v = DEB_p[5]
        self.kap = DEB_p[6]
        self.kap_R = DEB_p[7]
        self.p_M = DEB_p[8]
        self.p_T = DEB_p[9]
        self.k_J = DEB_p[10]
        self.E_G = DEB_p[11]
        self.E_Hb = DEB_p[12]
        self.E_Hp = DEB_p[13]
        self.T_A = DEB_p[14]
        self.T_ref = DEB_p[15]
        self.h_a = DEB_p[16]
        self.s_G = DEB_p[17]
        self.mu_X = DEB_p[18]
        self.mu_E = DEB_p[19]
        self.T_L = DEB_p[20]
        self.T_H = DEB_p[21]
        self.T_AL = DEB_p[22]
        self.T_AH = DEB_p[23]
        
        
        # DEB state variables #
        self.L = 0.0
        self.E = 0.0
        self.H = 0.0
        self.R = 0.0
        self.q = 0.0
        self.h = 0.0
        
    @property
    def DEB_p(self):
        return np.array([self.pAm_g_scatter, self.pAm_p_scatter, self.p_Am, self.F_m, self.kap_X, self.v, self.kap, self.kap_R, self.p_M, self.p_T, self.k_J, self.E_G, self.E_Hb, self.E_Hp, self.T_A, self.T_ref, self.h_a, self.s_G, self.mu_X, self.mu_E, self.T_L, self.T_H, self.T_AL, self.T_AH])
    
    @property
    def DEB_v(self):
        return np.array([self.L, self.E, self.H, self.R, self.q, self.h])
    
    @property
    def behave_p(self):
        return np.array([self.brood_size], dtype=np.float64)
    
    @DEB_v.setter
    def DEB_v(self, DEB_v):
        self.L = DEB_v[0]
        self.E = DEB_v[1]
        self.H = DEB_v[2]
        self.R = DEB_v[3]
        self.q = DEB_v[4]
        self.h = DEB_v[5]
    