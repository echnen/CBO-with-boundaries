import numpy as np 
import pickle 

import matplotlib.pyplot as plt

from cbx.utils.termination import * 

from hierarchicalNoiseObstacleCBO import * 

class pAllenCahnSolver: 

    def __init__(self, paco, glow, gup):

        self.p_allen_cahn_objective = paco 
        self.lower_obstacle_func = glow 
        self.upper_obstacle_func = gup       

    def __call__(self, n_iter=100, lamda=1, sigma=7, dt=0.01, seed=42, track_names=['consensus', 'energy', 'x'], os=1, caseID=""): 
    
        # set random seed 
        np.random.seed(seed)

        # initialize output 
        data = {} 

        # predefined spatial resolution 
        dx = 0.0078125
        n_points = int(round(1/dx)-1)
        
        n_particles = 20 * n_points  
        n_eval = n_particles * n_iter
        n_eval_final = 10 * n_eval

        self.batch_args = {
            'seed' : seed 
        }


        self.track_args = {'names':track_names, 'save_int': os}

        ########################################################################
        # Create and store obstacles 

        if self.lower_obstacle_func is not None: 
            g_lower_p = np.tile(self.lower_obstacle_func(np.linspace(0,1,n_points+2)[1:-1] ), (n_particles, 1))[None, ...]    
            data['lower_obstacle'] = g_lower_p[0,0,:]  
        else: 
            g_lower_p = None 

        if self.upper_obstacle_func is not None: 
            g_upper_p = np.tile(self.upper_obstacle_func(np.linspace(0,1,n_points+2)[1:-1] ), (n_particles, 1))[None, ...]    
            data['upper_obstacle'] = g_upper_p[0,0,:]
        else: 
            g_upper_p = None 

        ########################################################################
        # Initial particle distribution 

        bd_left = self.p_allen_cahn_objective.bd_left 
        bd_right = self.p_allen_cahn_objective.bd_right 

        # Initialize particles around the
        x = np.linspace(bd_left, bd_right, n_points)
        x = np.tile(x, (n_particles +1 , 1))[None, ...]
        x += hat_function_noise(32).sample(0.1*np.ones(x.shape))
        idx = np.argpartition(self.p_allen_cahn_objective(x), n_particles)
        x = x[:,idx[0,:n_particles],:]
        
        ########################################################################
        # Main Loop 
                
        for idx in range(7)[::-1]: 

            if idx == 0: 
                term_criteria = [max_eval_term(n_eval_final)]
            else: 
                term_criteria = [max_eval_term(n_eval)]

            my_noise = hat_function_noise(noise_every_x_points=2**idx, full_drift=False)
            
            dyn = hat_function_obstacleCBXDynamic(self.p_allen_cahn_objective, 
                                                  g_lower_p, 
                                                  g_upper_p, 
                                                  lamda, 
                                                  sigma, 
                                                  N=n_particles, 
                                                  x=x, 
                                                  dt=dt, 
                                                  term_criteria=term_criteria, 
                                                  track_args=self.track_args,
                                                  batch_args=self.batch_args, 
                                                  f_dim='3D', check_f_dims=False, 
                                                  noise = my_noise, 
                                                  noise_every_x_points=2**idx
                                                )

            best_particle = dyn.optimize()

            x *= 0
            x += dyn.x 

            data[idx] = {}
            for track_name in track_names: 
                data[idx]['hist_' + track_name] =   dyn.history[track_name]

        # create filename 
        filename = "p-allen-cahn/data/pAC"
        filename += '_ww_' + str(self.p_allen_cahn_objective.ww)
        filename += '_p_' + str(self.p_allen_cahn_objective.p) 
        filename += '_dx_' + str(dx)
        filename += '_dt_' + str(dt)
        filename += '_np_' + str(n_particles)
        filename += '_sg_' + str(sigma) 
        filename += '_lbd_' + str(lamda)
        filename += '_ne_' + str(n_eval) 
        filename += '_s_' + str(seed)
        filename += caseID
        filename += ".pickle"

        with open(filename, "wb") as output_file:
            pickle.dump(data, output_file)

        with open(filename, "rb") as input_file:
            e = pickle.load(input_file)

        return filename 
