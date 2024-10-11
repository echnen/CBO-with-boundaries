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
    plt.show()
    plt.close()


def plot_pAC_particles(filename): 
    with open(filename, "rb") as input_file:
        data = pickle.load(input_file)

    for idy in range(6):
        hist = data[idy]['hist_x']
        consensus = data[idy]['hist_consensus']
        
        plt.figure(figsize=(4,4),dpi=200) 
        x = np.linspace(0,1,129)
        for id_particle in range(hist[-1].shape[1]):     
            y = np.hstack([np.ones(1) * 0.5,hist[-1][0,id_particle,:],np.ones(1)])
            plt.plot(x,y, color ="gray", alpha = 0.4, linewidth=1)
        y = np.hstack([np.ones(1) * 0.5,consensus[-1][0,0,:],np.ones(1)])
        
        plt.plot(x[::2**idy], y[::2**idy], color = 'black', label="consensus", linewidth=2, marker="o", markersize=5)    
        
        plt.scatter(x[0],y[0],s=25,color = 'black', linewidth=2,zorder=20)
        plt.scatter(x[-1],y[-1],s=25,color = 'black', linewidth=2,zorder=20)

        plt.xlabel(r"$x$")
        plt.ylabel(r"$v(x)$")
        plt.xlim([-0.05,1.05])
        plt.ylim([0.4,1.1])
        # annotation of particles 
        plt.plot(x, x-15, color ="gray", label="particles", linewidth=2)
        plt.legend(loc="upper left", framealpha=1)

        plt.savefig("p-allen-cahn/images/pAC_particles_res" + str(7-idy) + ".pdf", bbox_inches = "tight")
        plt.close()

def plot_pAC_consensus_obstacles(filenames):
    
    for idf, filename in enumerate(filenames):  
        with open(filename, "rb") as input_file:
            data = pickle.load(input_file)

        plt.figure(figsize=(4,4),dpi=200) 

        plt.plot([-1,2], [0.75,0.75], color = "gray", linestyle = ":")
        plt.plot([-1,2], [0.25,0.25], color = "gray", linestyle = ":")

        hist = data[0]['hist_consensus']
        x = np.linspace(0,1,hist[-1].shape[2]+2)
        # TODO this should include the boundary conditions and not fixed values 
        y = np.hstack([np.ones(1)*0.5, hist[-1][0,0,:], np.ones(1)])
        plt.fill_between(x,-1,np.hstack([np.zeros(1), data["lower_obstacle"], np.zeros(1)]), color = "grey")
        plt.fill_between(x, 2,np.hstack([np.ones(1), data["upper_obstacle"], np.ones(1)]), color = "grey", label="obstacle")
        plt.plot(x,y,color="black",linewidth=2,label="consensus")

        plt.scatter(x[0],y[0],s=25,color="black",linewidth=2,zorder=20)
        plt.scatter(x[-1],y[-1],s=25,color="black",linewidth=2,zorder=20)

        plt.xlabel(r"$x$")
        plt.ylabel(r"$v(x)$")    
        plt.xlim([0.0,1.0])
        plt.ylim([-0.1,1.1])
        plt.legend(framealpha=1, loc = "upper left")
        plt.savefig("p-allen-cahn/images/pAC_consensus" + str(idf) + ".pdf", bbox_inches = 'tight') 
        plt.show()
        plt.close()

def plot_pAC_consensus_obstacles_p(filenames_p, parameters, pname, plabel, colormap): 
    cmap = matplotlib.colormaps[colormap]
    consensus = [] 

    for filename in filenames_p: 
        with open(filename, "rb") as input_file:
            data = pickle.load(input_file)
        consensus.append(data[0]['hist_consensus'][-1])

    plt.figure(figsize=(6,3),dpi=200) 
    plt.plot([-1,2],[0.25,0.25], color = "gray", linewidth=2, linestyle = ":")
    plt.plot([-1,2],[0.75,0.75], color = "gray", linewidth=2, linestyle = ":")

    x = np.linspace(0,1,consensus[0].shape[2]+2)
    plt.fill_between(x,-1,np.hstack([np.zeros(1), data["lower_obstacle"], np.zeros(1)]), color = "grey")
    plt.fill_between(x, 2,np.hstack([np.ones(1), data["upper_obstacle"], np.ones(1)]), color = "grey")

    for idp, p in enumerate(parameters): 
        color = cmap(0.1 + 0.8 * idp / (len(parameters)-1))
        y = np.hstack([np.ones(1) * 0.5,consensus[idp][0,0,:],np.ones(1)])
        plt.plot(x,y,color=color, linewidth=2, label = plabel + str(parameters[idp]))

    plt.scatter(x[0],y[0],color = cmap(0.9), linewidth=2, zorder=20)
    plt.scatter(x[-1],y[-1],color = cmap(0.9), linewidth=2, zorder=20)

    plt.xlabel(r"$s$")
    plt.ylabel(r"$v(s)$")    

    plt.xlim([-0.0,1.0])
    plt.ylim([-0.05,1.1])

    plt.legend(framealpha = 1, loc = "upper left", ncol=4)
    plt.savefig("p-allen-cahn/images/pAC_consensus_" + pname + ".pdf",bbox_inches="tight")
    plt.show()


def plot_pAC_energies_obstacles_p(filenames_p, parameters, pname, plabel, colormap): 

    energies = [] 

    for filename in filenames_p: 
        with open(filename, "rb") as input_file:
            data = pickle.load(input_file)

        energies.append([np.abs(np.asarray(data[idx]['hist_energy']) - np.min(data[idx]['hist_energy'][1:])) for idx in range(6)]) 
    iterations = energies[0][-1].shape[0]
    

    cmap = matplotlib.colormaps[colormap]
    greys = matplotlib.colormaps['Greys']
    
    plt.figure(figsize=(6,3),dpi=200) 

    ylim=[1e-5,1e4]

    start = 0 
    end = 0 
    for idx in range(6)[::-1]:
        end += iterations 
        plt.fill_between(np.arange(start,end+1), ylim[0], ylim[1], alpha=0.5, color = greys(idx/9))
        if idx != 5:    
            plt.plot([start,start], ylim, color = "gray", linestyle = ":")
        start += iterations

    for idp, p in enumerate(parameters): 
        color = cmap(0.1 + 0.8 * idp / (len(parameters)-1))
        start = 0 
        end = 0  
        x = np.asarray([0])
        y = np.asarray([0])
        for idy in range(1,6)[::-1]:
            end += iterations 
            plt.plot(np.arange(start,end),energies[idp][idy][:,0],linewidth=2, color = color)
            start += iterations
        plt.plot(np.arange(start,end+2*iterations), energies[idp][0][::5,0],linewidth=2, 
                 color = color, label = plabel + str(p))


    plt.xlim([0,7 * iterations])
    plt.ylim(ylim)

    plt.xticks(ticks = [x * iterations for x in range(8)], labels= [x * iterations for x in range(6)]+["...",15*iterations])
    plt.yscale('log')

    plt.xlabel("Iterations")
    plt.ylabel(r"Energy - min Energy of the $i$th run")

    plt.legend(loc = "upper left", framealpha=1, ncols=4) 

    plt.savefig("p-allen-cahn/images/pAC_energies_obstacle_" + pname+ ".pdf",bbox_inches="tight")
    plt.show()