from objectives import p_allen_cahn_objective
from obstacles import *
from pAllenCahnSolver import pAllenCahnSolver

def exp1_p_Allen_Cahn(): 
    paco = p_allen_cahn_objective(ww = 500, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    # no lower obstacle, no upper obstacle 
    solver = pAllenCahnSolver(paco=paco, glow=None, gup=None)
    return solver()

def exp2_p_Allen_Cahn_obstacles(): 
    paco = p_allen_cahn_objective(ww = 500, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    
    solverA = pAllenCahnSolver(paco=paco, glow=lower_geometric_func2, gup=half_upper_sin_func)
    solverB = pAllenCahnSolver(paco=paco, glow=lower_sides_geometric_func, gup=upper_sin_func)
    solverC = pAllenCahnSolver(paco=paco, glow=lower_geometric_func2, gup=upper_sin_func)

    track_names = ['consensus']
    filenameA = solverA(n_iter=1000, track_names=track_names, os=100, caseID="A")
    filenameB = solverB(n_iter=1000, track_names=track_names, os=100, caseID="B")
    filenameC = solverC(n_iter=1000, track_names=track_names, os=100, caseID="C")

    return [filenameA, filenameB, filenameC]
    
def exp3_p_Allen_Cahn_obstacle_ps(): 
    paco_p1 =   p_allen_cahn_objective(ww = 2000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1)
    paco_p125 = p_allen_cahn_objective(ww = 2000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.25)
    paco_p15 =  p_allen_cahn_objective(ww = 2000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    paco_p2 =   p_allen_cahn_objective(ww = 2000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=2)

    solver_p1 =   pAllenCahnSolver(paco=paco_p1, glow=lower_geometric_func2, gup=upper_sin_func)
    solver_p125 = pAllenCahnSolver(paco=paco_p125, glow=lower_geometric_func2, gup=upper_sin_func)
    solver_p15 =  pAllenCahnSolver(paco=paco_p15, glow=lower_geometric_func2, gup=upper_sin_func)
    solver_p2 =   pAllenCahnSolver(paco=paco_p2, glow=lower_geometric_func2, gup=upper_sin_func)

    track_names = ['consensus', 'energy']
    filename_p1 =   solver_p1(n_iter=1000, track_names=track_names, os=1, caseID="p1")
    filename_p125 = solver_p125(n_iter=1000, track_names=track_names, os=1, caseID="p1.25")
    filename_p15 =  solver_p15(n_iter=1000, track_names=track_names, os=1, caseID="p1.5")
    filename_p2 =   solver_p2(n_iter=1000, track_names=track_names, os=1, caseID="p2")
    
    return [filename_p1, filename_p125, filename_p15, filename_p2]

def exp4_p_Allen_Cahn_obstacle_wws(): 
    paco_ww500 =   p_allen_cahn_objective(ww = 500, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    paco_ww2000 =  p_allen_cahn_objective(ww = 2000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    paco_ww10000 = p_allen_cahn_objective(ww = 10000, w1=0.25, w2=0.75, bd_left=0.5, bd_right=1, p=1.5)
    
    solver_ww500 =   pAllenCahnSolver(paco=paco_ww500, glow=lower_geometric_func2, gup=upper_sin_func)
    solver_ww2000 =  pAllenCahnSolver(paco=paco_ww2000, glow=lower_geometric_func2, gup=upper_sin_func)
    solver_ww10000 = pAllenCahnSolver(paco=paco_ww10000, glow=lower_geometric_func2, gup=upper_sin_func)
    
    track_names = ['consensus', 'energy']
    filename_ww500 = solver_ww500(n_iter=1000, track_names=track_names, os=1, caseID="ww500")
    filename_ww2000 = solver_ww2000(n_iter=1000, track_names=track_names, os=1, caseID="ww2000")
    filename_ww10000 = solver_ww10000(n_iter=1000, track_names=track_names, os=1, caseID="ww10000")
    
    return [filename_ww500, filename_ww2000, filename_ww10000]