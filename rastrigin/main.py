# -*- coding: utf-8 -*-
#
#    Copyright (C) 2024 Jonas Beddrich (beddrich@ma.tum.de)
#                       Enis Chenchene (enis.chenchene@univie.ac.at)
#                       Massimo Fornasier (massimo.fornasier@cit.tum.de)
#                       Hui Huang (hui.huang@uni-graz.at)
#                       Barbara Wohlmuth (wohlmuth@ma.tum.de)
#
#    This file is part of the example code repository for the paper:
#
#      J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
#      Constrained Consensus-Based Optimization and Numerical Heuristics for
#      the Low Particle Regime,
#      2024. DOI: 10.48550/arXiv.2410.10361.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Run this file to reproduce all the experiments in Section 5.1 and 5.2 of:

J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
Constrained Consensus-Based Optimization and Numerical Heuristics for the Low
Particle Regime,
2024. DOI: 10.48550/arXiv.2410.10361.

"""

import pathlib
import experiments as expm
import cbo as opt
import numpy as np
import plots as show


if __name__ == "__main__":

    pathlib.Path("Results").mkdir(parents=True, exist_ok=True)

    print("------------------------------------------------------------------")
    print("\n\n *** Experiment 1 ***\n")
    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 2 ***\n")

    # dom = opt.Domain('Box', np.array([-6.12, 5.12]))
    # Out0 = expm.parameter_test_multiple(100, 100, dom, noise_type='isotropic')
    # Out2 = expm.parameter_test_multiple(100, 100, dom, noise_type='anisotropic')

    # dom = opt.Domain('unconstrained')
    # Out1 = expm.parameter_test_multiple(100, 100, dom, noise_type='isotropic')
    # Out3 = expm.parameter_test_multiple(100, 100, dom, noise_type='anisotropic')

    show.plot_testing_boundedness_iso(Out0[0], Out0[1], [Out0[2], Out1[2]],
                                      [Out0[3], Out1[3]],
                                    'Results/testing_boundedness_isotropic.pdf')
    show.plot_testing_boundedness_ani(Out0[0], Out0[1], [Out2[2], Out3[2]],
                                      [Out2[3], Out3[3]],
                                  'Results/testing_boundedness_anisotropic.pdf')

    print("------------------------------------------------------------------")
    print("\n\n *** Experiment 2 ***\n")
    # show.plot_testing_boundedness_iso(Out0[0], Out0[1], [Out0[2], Out1[2]],
    #                                   [Out0[3], Out1[3]],
    #                                'Results/testing_boundedness_isotropic.pdf')
    # show.plot_testing_boundedness_ani(Out0[0], Out0[1], [Out2[2], Out3[2]],
    #                                   [Out2[3], Out3[3]],
    #                              'Results/testing_boundedness_anisotropic.pdf')

    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 3 ***\n")

    # lam = 1
    # sig = 5 * np.sqrt(2 * lam)
    # dom = opt.Domain('Box', np.array([0, 11.24]))
    # Out4 = expm.optimize_middle_scale(lam, sig, dom)
    # show.plot_experiment_large_scale_many(*Out4,
    #                             'Results/middle_scale_on_boundary_not_feas.pdf')
    # dom = opt.Domain('Box', np.array([-6.12, 5.12]))
    # Out5 = expm.optimize_middle_scale(lam, sig, dom)
    # show.plot_experiment_large_scale_many(*Out5,
    #                             'Results/middle_scale_in_interior_not_feas.pdf')

    # lam = 1
    # sig = np.sqrt(2 * lam)
    # dom = opt.Domain('Box', np.array([0, 11.24]))
    # Out6 = expm.optimize_middle_scale(lam, sig, dom)
    # show.plot_experiment_large_scale_many(*Out6,
    #                                 'Results/middle_scale_on_boundary_feas.pdf')
    # dom = opt.Domain('Box', np.array([-6.12, 5.12]))
    # Out7 = expm.optimize_middle_scale(lam, sig, dom)
    # show.plot_experiment_large_scale_many(*Out7,
    #                                 'Results/middle_scale_in_interior_feas.pdf')


    print("------------------------------------------------------------------")
    print("\n\n *** Experiment 3 ***\n")

    dom = opt.Domain('unconstrained')
    Vs_tot, dt, maxit, Ss = expm.testing_heuristics(dom)
    show.plot_experiment_heuristics(Vs_tot, dt, maxit, Ss)

    print("------------------------------------------------------------------")
    print("\n\n *** Experiment 4 ***\n")
    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 4 ***\n")

    # dom = opt.Domain('Box', np.array([0, 11.24]))
    # Out8 = expm.optimize_large_scale(1000, dom)
    # show.plot_residuals(Out8[0], Out8[1], 'Results/large_scale_bounded.pdf')

    # dom = opt.Domain('unconstrained')
    # Out9 = expm.optimize_large_scale(10000, dom)
    # show.plot_residuals(Out9[0], Out9[1], 'Results/large_scale_unbounded.pdf')

    dom = opt.Domain('unconstrained')
    Out9bis = expm.optimize_large_scale_standard(10000, dom)
    show.plot_residuals(Out9bis[0], Out9bis[1],
                        'Results/large_scale_unbounded_standard.pdf')

    print("------------------------------------------------------------------")
    print("\n\n *** Experiment 5 ***\n")
    # print("------------------------------------------------------------------")
    # print("\n\n *** Generating Figure 5 ***\n")

    # Out9 = expm.experiment_non_convex_domain('noncvx1')
    # show.plot_experiment_large_scale_many(*Out9, 'Results/nncvx_domain_1.pdf')

    # Out10 = expm.experiment_non_convex_domain('noncvx2')
    # show.plot_experiment_large_scale_many(*Out10, 'Results/nncvx_domain_2.pdf')

    # Out11 = expm.experiment_number_particles()
    # show.plot_experiment_number_particles(*Out11, 'Results/number_particles.pdf')

    Out12 = expm.experiment_number_particles_comparison_constrained()
    show.plot_experiment_number_particles_comparison(*Out12, 'Results/number_particles_comparison_constrained.pdf')
