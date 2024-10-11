from objectives import p_allen_cahn_objective
from obstacles import *
from pAllenCahnSolver import pAllenCahnSolver

def exp1_p_Allen_Cahn(): 
    paco = p_allen_cahn_objective(ww = 500, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    # no lower obstacle 
    # no upper obstacle 
    solver = pAllenCahnSolver(paco=paco, glow=None, gup=None)
    return solver()

def exp2_p_Allen_Cahn_obstacles(): 
    print()

def exp3_p_Allen_Cahn_obstacle_parameterss(): 
    print()