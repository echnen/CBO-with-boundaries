import numpy as np 

class p_allen_cahn_objective: 
    
    def __init__(self, ww=10000, w1=0.25, w2=0.75, bd_left = 0, bd_right = 0, p=2): 
        self.ww = ww
        self.w1 = w1
        self.w2 = w2 
        self.bd_left = bd_left
        self.bd_right = bd_right
        self.p = p 

    def __call__(self,v): 

        n_particles = v.shape[1]
        n_points = v.shape[2]
        dx = 1/(n_points+1)

        v_p = np.c_[np.ones(n_particles) * self.bd_left, v[0], np.ones(n_particles) * self.bd_right]
        v_p = v_p[None, ...]

        dvdx = (v_p[...,1:] - v_p[...,:-1]) / dx
        dwp = self.ww * (v_p - self.w1) ** 2 * (v_p - self.w2) ** 2 

        return np.sum(np.abs(dvdx) ** self.p / self.p, axis=2) + np.sum(dwp, axis=2)