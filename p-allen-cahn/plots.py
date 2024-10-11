import matplotlib.pyplot as plt 
import matplotlib
import numpy as np 
import time 
from scipy import interpolate

from cbx.utils.termination import * 
from cbx.utils.resampling import * 
from cbx.scheduler import scheduler, multiply

import pickle 

import matplotlib
import matplotlib.colors as colors 
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_pAC_energy(filename):

    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)

    # Get minimal energy 
    energy_min = data[0]['hist_energy'][-1]
    energies = {}
    for idx in range(6):
        energies[idx] = np.abs(np.asarray(data[idx]['hist_energy']) - energy_min)

    cmap = matplotlib.colormaps['viridis_r']

    plt.figure(figsize=(6,4),dpi=200) 

    y_lim=[1e-3,1e3]

    start = 0 
    end = 0  
    for idx in range(6)[::-1]:
        end += energies[idx].shape[0]
        plt.fill_between(np.arange(start,end+1), y_lim[0], y_lim[1], alpha=0.25, color = cmap(idx/7+1/7))
        if idx != 5: 
            plt.plot([start,start], y_lim, color = "gray", linestyle = ":")
        plt.plot(np.arange(start,end), energies[idx], color = cmap(idx/7+1/7), linewidth=2, label = str(7-idx))
        start += energies[idx].shape[0]

    plt.yscale('log')
    plt.xlim([0,1500])
    plt.ylim(y_lim)
    plt.xlabel("Iterations")
    plt.ylabel("Energy - minimal Energy")
    plt.legend(ncols=3, framealpha=1)

    plt.savefig("p-allen-cahn/images/pAC_energies.pdf", bbox_inches = 'tight') 
    plt.close()

def plot_pAC_consensus(filename): 
    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)

    plt.figure(figsize=(6,4),dpi=200) 

    plt.plot([-1,2], [0.75,0.75], color = "gray", linestyle = ":")
    plt.plot([-1,2], [0.25,0.25], color = "gray", linestyle = ":")

    cmap = matplotlib.colormaps['viridis_r']

    for idx in range(6)[::-1]:

        hist = data[idx]['hist_consensus']
        x = np.linspace(0,1,hist[-1].shape[2]+2)
        y = np.hstack([np.ones(1) * 0.5,hist[-1][0,0,:],np.ones(1)])
        plt.plot(x,y, color = cmap(idx/7+1/7), linewidth=2, label=7-idx)

    plt.scatter(x[0],y[0],s=25,color = cmap(idx/7+1/7), linewidth=2,zorder=20)
    plt.scatter(x[-1],y[-1],s=25,color = cmap(idx/7+1/7), linewidth=2,zorder=20)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$v(x)$")    
    plt.xlim([-0.05,1.05])
    plt.ylim([0.4,1.1])
    plt.legend(ncols=3, framealpha=1)
    plt.savefig("p-allen-cahn/images/pAC_consensus.pdf", bbox_inches = 'tight') 
    plt.close()


# def plot_pAC_particles(): 
#     with open("data/hat_function_noise/data_ww_500.0_p_1.5_sc_0_ne_254000_np_2540_s_42.pickle", "rb") as input_file:
#         data = pickle.load(input_file)

#     sc = data['config']['scenario']
#     ww = data['config']['well weight']
#     os = data['config']['output steps']

#     for idy in range(6):
#         hist = data[idy]['hist_x']
#         consensus = data[idy]['hist_consensus']
#         cmap = matplotlib.colormaps['viridis_r']
        
#         plt.figure(figsize=(4,4),dpi=200) 
#         x = np.linspace(0,1,129)
#         for id_particle in range(hist[-1].shape[1]):     
#             y = np.hstack([np.ones(1) * 0.5,hist[-1][0,id_particle,:],np.ones(1)])
#             plt.plot(x,y, color ="gray", alpha = 0.4, linewidth=1)
#         y = np.hstack([np.ones(1) * 0.5,consensus[-1][0,0,:],np.ones(1)])
        
#         plt.plot(x[::2**idy], y[::2**idy], color = 'black', label="consensus", linewidth=2, marker="o", markersize=5)    
        
#         plt.scatter(x[0],y[0],s=25,color = 'black', linewidth=2,zorder=20)
#         plt.scatter(x[-1],y[-1],s=25,color = 'black', linewidth=2,zorder=20)

#         plt.xlabel(r"$x$")
#         plt.ylabel(r"$v(x)$")
#         plt.xlim([-0.05,1.05])
#         plt.ylim([0.4,1.1])
#         # annotation of particles 
#         plt.plot(x, x-15, color ="gray", label="particles", linewidth=2)
#         plt.legend(loc="upper left", framealpha=1)
#         plt.savefig("AC_sc_"+str(sc)+"_ww_" + str(ww)+"_final_particle distribution"+ str(idy) +".pdf", bbox_inches = "tight")