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
This file contains an implementation of the experiments conducted in Sections
5.1 and 5.2 of the paper:

J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
Constrained Consensus-Based Optimization and Numerical Heuristics for the Low
Particle Regime,
2024. DOI: 10.48550/arXiv.2410.10361.

"""

import numpy as np
import cbo as opt
from tqdm import trange
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


def parameter_test_multiple(num_of_runs, maxit, dom, noise_type='isotropic'):
    '''
    We compare different choices of the parameters
    '''

    # model parameters
    d = 2
    x_opt = np.zeros(d)

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    N = 1000
    gamma = 1

    Lambdas = np.logspace(-1, 2.5, num_of_runs)
    Sigmas = np.logspace(-1, 1.1, num_of_runs)

    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, num_of_runs))

    # starting point
    mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
    variance = 1e1
    X0 = dom.proj(np.random.normal(mean, variance, (d, N)))
    V0 = np.sum((X0 - x_opt[:, np.newaxis]) ** 2) / N

    for i in trange(num_of_runs, desc="Generating image", leave=False):
        for j in range(num_of_runs):

            lam = Lambdas[i]
            sig = Sigmas[j]

            # optimizing
            Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt,
                         maxit, noise_type, gamma, dom, function='Rastrigin',
                         Heating=False, Ball=False, Verbose=False)
            Vs_tot[i, j] = min(max(np.log(Vs[-1]), -10), 8)

    return Lambdas, Sigmas, Vs_tot, V0


def optimize_large_scale(maxit, dom):
    '''
    We try to optimize the Rastrigin function in high dimensions.
    '''
    num_of_runs = 20

    # model parameters
    d = 100
    x_opt = np.zeros(d)

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    N = 10000

    # algorithms parameters
    lam = 1
    sig = 5 * np.sqrt(2 * lam)
    gamma = .95

    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, maxit))

    for it in trange(num_of_runs, desc="Optimizing", leave=False):

        # starting point
        mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
        variance = 1e1
        X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

        # optimizing
        Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt,
                     maxit, 'anisotropic', gamma, dom, function='Rastrigin',
                     Heating=True, Ball=True, Plot=False, Verbose=True)

        Vs_tot[it, :] = Vs

    return Vs_tot, dt, maxit


def optimize_middle_scale(lam, sig, dom):
    '''
    We try to optimize the Rastrigin function in middle-scale dimensions for
    several parameters
    '''

    num_of_runs = 20
    maxit = 1000

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    N = 1000
    Gammas = [1, 0.75, 0.5]
    num_gammas = len(Gammas)

    # testing the following dimensions
    Ds = [2, 15, 20]
    num_ds = len(Ds)

    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, maxit, num_ds, num_gammas))

    for (cnt_d, d) in enumerate(Ds):
        for (cnt_gamma, gamma) in enumerate(Gammas):

            # optimal solution
            x_opt = np.zeros(d)

            for t in trange(num_of_runs):

                # starting point
                mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
                variance = 1e1
                X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

                # optimizing
                Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt,
                             maxit, 'anisotropic', gamma, dom,
                             function='Rastrigin',
                             Heating=True, Ball=True, Verbose=False)

                Vs_tot[t, :, cnt_d, cnt_gamma] = Vs

    return Vs_tot, dt, maxit, Ds, Gammas


def experiment_non_convex_domain(dom_name):

    num_of_runs = 20
    maxit = 1000

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    N = 1000
    Gammas = [1, 0.75, 0.5]
    num_gammas = len(Gammas)
    lam = 1
    sig = 5 * np.sqrt(2 * lam)

    # testing the following dimensions
    Ds = [2, 15, 20]
    num_ds = len(Ds)

    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, maxit, num_ds, num_gammas))

    for (cnt_d, d) in enumerate(Ds):
        for (cnt_gamma, gamma) in enumerate(Gammas):

            # optimal solution and domain
            x_opt = np.zeros(d)
            dom = opt.Domain(dom_name, [np.zeros(d), 5.12, d])

            for t in trange(num_of_runs):

                # starting point
                mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
                variance = 1e1
                X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

                Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
                             'anisotropic', gamma, dom, function='Rastrigin',
                             Heating=True, Ball=True, Verbose=False)

                Vs_tot[t, :, cnt_d, cnt_gamma] = Vs

    return Vs_tot, dt, maxit, Ds, Gammas


def experiment_number_particles():

    num_of_runs = 20
    maxit = 1000

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    Ns = [10,20,30]
    num_ns = len(Ns)
    Gammas = [1, 0.75]
    num_gammas = len(Gammas)
    lam = 1
    sig = 5 * np.sqrt(2 * lam)

    # testing the following dimensions
    d = 20
    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, maxit, num_gammas + 1, num_ns))

    for (cnt_gamma, gamma) in enumerate(Gammas):
        for (cnt_N, N) in enumerate(Ns):

            # optimal solution and domain
            x_opt = np.zeros(d)
            dom = opt.Domain('Box', np.array([-6.12, 5.12]))

            for t in trange(num_of_runs):

                # starting point
                mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
                variance = 1e1
                X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

                Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
                            'anisotropic', gamma, dom, function='Rastrigin',
                            Heating=True, Ball=True, Verbose=False)

                Vs_tot[t, :, cnt_gamma, cnt_N] = Vs

    sig = np.sqrt(2 * lam)

    for (cnt_N, N) in enumerate(Ns):

        # optimal solution and domain
        x_opt = np.zeros(d)
        dom = opt.Domain('unconstrained', np.array([-6.12, 5.12]))

        for t in trange(num_of_runs):

            # starting point
            mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
            variance = 1e1
            X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

            Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
                        'anisotropic', gamma, dom, function='Rastrigin',
                        Heating=True, Ball=True, Verbose=False)

            Vs_tot[t, :, num_gammas, cnt_N] = Vs

    return Vs_tot, dt, maxit, d, Gammas + ["CBO"], Ns


def experiment_number_particles_comparison_constrained():

    def rastrigin_PSO(x):
        d = x.shape[1]
        return 10.0 * d + (x ** 2.0 - 10.0 * np.cos(2.0 * np.pi * x)).sum(axis=1)

    num_of_runs = 20
    maxit = 1000

    # algorithm parameters
    dt = 1e-2
    alpha = 1e6
    Ns = 2 ** np.arange(15)
    num_ns = len(Ns)
    Gammas = [1.0, 0.75]
    num_gammas = len(Gammas)
    lam = 1
    sig = 5 * np.sqrt(2 * lam)

    # testing the following dimensions
    Ds = [5,10,15,20]
    num_Ds = len(Ds)

    # placeholders for plots
    Vs_tot = np.zeros((num_of_runs, maxit, num_Ds, num_gammas + 2, num_ns))

    for (cnt_d, d) in enumerate(Ds):  
        for (cnt_gamma, gamma) in enumerate(Gammas):
            for (cnt_N, N) in enumerate(Ns):

                # optimal solution and domain
                x_opt = np.zeros(d)
                dom = opt.Domain('Box', np.array([-6.12, 5.12]))

                for t in trange(num_of_runs):

                    # starting point
                    mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
                    variance = 1e1
                    X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

                    Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
                                'anisotropic', gamma, dom, function='Rastrigin',
                                Heating=True, Ball=True, Verbose=False)

                    Vs_tot[t, :, cnt_d, cnt_gamma, cnt_N] = Vs

    sig = np.sqrt(2 * lam)

    for (cnt_d, d) in enumerate(Ds):  
        for (cnt_N, N) in enumerate(Ns):

            # optimal solution and domain
            x_opt = np.zeros(d)
            dom = opt.Domain('Box', np.array([-6.12, 5.12]))

            for t in trange(num_of_runs):

                # starting point
                mean = 5.12 * np.ones((d, N)) / np.sqrt(d)
                variance = 1e1
                X0 = dom.proj(np.random.normal(mean, variance, (d, N)))

                Vs = opt.cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
                            'anisotropic', gamma, dom, function='Rastrigin',
                            Heating=True, Ball=True, Verbose=False)

                Vs_tot[t, :, cnt_d, num_gammas, cnt_N] = Vs

    for (cnt_d, d) in enumerate(Ds):  
        for (cnt_N, N) in enumerate(Ns):
            for t in trange(num_of_runs):

                # bounded PSO 
                max_bound = 5.12 * np.ones(d)
                min_bound = -6.12 * np.ones(d)
                bounds = (min_bound, max_bound)
                # swarm initialization 
                options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

                optimizer = ps.single.GlobalBestPSO(n_particles=N, dimensions=d, options=options, bounds=bounds)
                cost, pos = optimizer.optimize(rastrigin_PSO, iters=maxit)
                ws_distance = np.sum(np.asarray(optimizer.pos_history)**2, axis=(1,2)) / N
                Vs_tot[t, :, cnt_d, num_gammas+1, cnt_N] = ws_distance

    return Vs_tot, dt, maxit, Ds, Gammas + ["CBO", "PSO"], Ns
