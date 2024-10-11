from cbx.dynamics import CBXDynamic
from typing import Callable
from numpy.typing import ArrayLike
from numpy.random import normal 
from numpy.random import laplace
import matplotlib.pyplot as plt 
import numpy as np 


class hat_function_obstacleCBXDynamic(CBXDynamic): 

    def __init__(self, f, lower_obstacle=None, upper_obstacle=None, lamda = 1, sigma=1, noise_every_x_points = 1, bd_left=0.5, bd_right=1, **kwargs) -> None:
        super().__init__(f, **kwargs)

        # problem setting 
        if lower_obstacle is not None: 
            self.lower_obstacle = lower_obstacle
        if upper_obstacle is not None: 
            self.upper_obstacle = upper_obstacle
        self.bd_left = bd_left
        self.bd_right = bd_right
        
        # algorithmic parameters 
        self.lamda = lamda
        self.sigma = sigma
        self.nexp = noise_every_x_points
        
    def inner_step(self,) -> None:
        # create interpolation_matrix 
        if not hasattr(self, 'IM'):
            self.IM = self.create_interpolation_matrix(self.nexp, self.x.shape[2]+2)

        # compute consensus point and energy
        self.consensus, energy = self.compute_consensus()
        self.energy[self.consensus_idx] = energy

        # compute drift         
        self.drift = self.x[self.particle_idx] - self.consensus

        # compute noise
        self.s = self.sigma * self.noise()

        # update particle positions
        self.x[self.particle_idx] = (
            self.x[self.particle_idx] -
            self.correction(self.lamda * self.dt * self.drift) +
            self.s)

        # project particles onto domain     
        if hasattr(self, 'lower_obstacle'): 
            idx = self.x < self.lower_obstacle 
            self.x[idx] = self.lower_obstacle[idx]   
        if hasattr(self, 'upper_obstacle'): 
            idx = self.x > self.upper_obstacle
            self.x[idx] = self.upper_obstacle[idx]
        if hasattr(self, 'lower obstacle') or hasattr(self, 'upper obstacle'): 
            # Resolving the obstacle at the same resolution as the noise and solution 
            self.project_x_to_coarse()

    def project_x_to_coarse(self,) -> None:
        left  = np.ones((self.x.shape[0], self.x.shape[1], 1)) * self.bd_left
        right = np.ones((self.x.shape[0], self.x.shape[1], 1)) * self.bd_right
        bd_x = np.concatenate((left, self.x[:,:,self.nexp-1::self.nexp], right), axis=2)

        IMxT = np.matmul(self.IM, np.transpose(bd_x, (0,2,1)))
        self.x = np.transpose(IMxT, (0,2,1))[:,:,1:-1]

    def create_interpolation_matrix(self, nexp, dimension):    
        # nexp :=  noise every x points should be a power of two 
        # x . . . x = 3
        # with boundary 
        # dimension should be a power of two +1 
        
        v = (np.hstack([np.arange(nexp), np.arange(nexp)[:-1][::-1]]) + 1)  / (nexp)
        v = v.reshape(v.shape[0],1)

        # first column, initial creation of IM to hstack afterwards 
        IM = np.vstack([v[nexp-1:], np.zeros((dimension - nexp,1))])
        # core of IM 
        for idx in range(0, int(dimension / nexp)-1):
            column = np.vstack([np.zeros((1 + idx * nexp, 1)), v, np.zeros((dimension - (idx+2) * nexp , 1))])
            IM = np.hstack([IM, column])
        if nexp > 1: 
            column = np.vstack([np.zeros((dimension - nexp,1)), v[:nexp]])
            IM = np.hstack([IM, column])
        return IM
    
class hat_function_noise:
        
    def __init__(self,
                 noise_every_x_points,  
                 full_drift = False, 
                 norm: Callable = None,
                 sampler: Callable = None
                 ):
                
        self.norm = norm if norm is not None else np.linalg.norm
        self.sampler = sampler if sampler is not None else normal
        self.nexp = noise_every_x_points
        self.full_drift = full_drift
    
    def __call__(self, dyn) -> ArrayLike:
        return np.sqrt(dyn.dt) * self.sample(dyn.drift)
    
    def sample(self, drift: ArrayLike) -> ArrayLike:
        n_runs, n_particles, dimension = drift.shape
        # print('dim', dimension, 'nexp', self.nexp)
        
        sample = np.zeros((n_runs,n_particles,dimension))
        r = self.sampler(0,1,(n_runs, n_particles, int(dimension / self.nexp)))
        # print("Hi", r.shape, self.nexp)
        if not self.full_drift: 
            r *= drift[:,:,self.nexp-1::self.nexp]

        IM = self.create_interpolation_matrix(self.nexp,dimension)[None,:]
        IMrT = np.matmul(IM, np.transpose(r,(0,2,1)))
        sample = np.transpose(IMrT, (0,2,1))
    
        if self.full_drift: 
            sample *= drift
        
        return sample 
    
    def create_interpolation_matrix(self, nexp, dimension):    
        v = (np.hstack([np.arange(nexp), np.arange(nexp)[:-1][::-1]]) + 1)  / (nexp)
        v = v.reshape(v.shape[0],1)

        # first column, initial creation of IM to hstack afterwards 
        IM = np.vstack([v, np.zeros((dimension - 2 * nexp + 1 ,1))])
        # core of IM 
        for idx in range(1, int(dimension / nexp)-1):
            column = np.vstack([np.zeros((idx * nexp , 1)), v, np.zeros((dimension - idx * nexp - 2 * nexp + 1, 1))])
            IM = np.hstack([IM, column])
        # last column 
        last_column_entries = dimension - (int(dimension / nexp) * nexp) + nexp
        if last_column_entries < dimension: 
            column = np.vstack([np.zeros(((dimension-last_column_entries),1)), v[:last_column_entries]])
            IM = np.hstack([IM, column])
        return IM

if __name__ == '__main__': 

    n_max = 8

    plt.figure()
    plt.title("No drift")
    one_drift = np.ones((1,1,2**n_max-1))
    for i in range(n_max):   
        my_noise = hat_function_noise(2**i) 
        sample = my_noise.sample(one_drift)
        plt.plot([0] + list(sample[0,0,:]) + [0], label = str(i))
    plt.legend()

    # drift for the following two 
    random_drift = np.random.normal(0,1,(1,1,2**n_max-1))

    plt.figure()
    plt.title("Normally distributed drift - applied to random values")
    for i in range(n_max): 
        my_noise = hat_function_noise(2**i, full_drift=False) 
        sample = my_noise.sample(random_drift)
        plt.plot([0] + list(sample[0,0,:]) + [0], label = str(i))
    plt.legend()

    plt.figure()
    plt.title("Normally distributed drift - applied to full sample")
    for i in range(n_max): 
        my_noise = hat_function_noise(2**i, full_drift=True) 
        sample = my_noise.sample(random_drift)
        plt.plot([0] + list(sample[0,0,:]) + [0], label = str(i))
    plt.legend()

    plt.show() 