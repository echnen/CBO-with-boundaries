from experiments import * 
from plots import *

if __name__ == '__main__': 

    filename = "p-allen-cahn/data/pAC_ww_500_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_254000_s_42.pickle"
    # filename = exp1_p_Allen_Cahn()
    # plot_pAC_energy(filename)
    # plot_pAC_consensus(filename)
    # plot_pAC_particles(filename)
    
    # experiment 2 
    filenames = ["p-allen-cahn/data/pAC_ww_500_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42A.pickle", 
                 "p-allen-cahn/data/pAC_ww_500_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42B.pickle", 
                 "p-allen-cahn/data/pAC_ww_500_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42C.pickle"]
    # filenames = exp2_p_Allen_Cahn_obstacles()
    plot_pAC_consensus_obstacles(filenames)

    # experiment 3
    filenames_p = ["p-allen-cahn/data/pAC_ww_2000_p_1_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42p1.pickle", 
                   "p-allen-cahn/data/pAC_ww_2000_p_1.25_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42p1.25.pickle", 
                   "p-allen-cahn/data/pAC_ww_2000_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42p1.5.pickle", 
                   "p-allen-cahn/data/pAC_ww_2000_p_2_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42p2.pickle"]

    # filenames_p = exp3_p_Allen_Cahn_obstacle_ps()
    # plot_pAC_consensus_obstacles_p(filenames_p, parameters=[1, 1.25, 1.5, 2], pname = "p", plabel=r"$p = $", colormap = 'plasma')
    # plot_pAC_energies_obstacles_p(filenames_p, parameters=[1, 1.25, 1.5, 2], pname = "p", plabel=r"$p = $", colormap = 'plasma')

    # experiment 4
    filenames_ww = ["p-allen-cahn/data/pAC_ww_500_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42ww500.pickle", 
                   "p-allen-cahn/data/pAC_ww_2000_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42ww2000.pickle", 
                   "p-allen-cahn/data/pAC_ww_10000_p_1.5_dx_0.0078125_dt_0.01_np_2540_sg_7_lbd_1_ne_2540000_s_42ww10000.pickle"]

    # filenames_ww = exp4_p_Allen_Cahn_obstacle_wws()
    # plot_pAC_consensus_obstacles_p(filenames_ww, parameters=[500,2000,10000], pname = "ww", plabel=r"$\epsilon^{-2} = $", colormap = 'inferno')
    # plot_pAC_energies_obstacles_p(filenames_ww, parameters=[500,2000,10000], pname = "ww", plabel=r"$\epsilon^{-2} = $", colormap = 'inferno')

    filenames_ww = []  

