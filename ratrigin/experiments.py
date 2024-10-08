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
#      Constrained Consensus-Based Optimization and Numerical Heuristics for the Low Particle Regime,
#      2024. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.
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
This file contains an implementation of the experiments conducted in:

J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
Constrained Consensus-Based Optimization and Numerical Heuristics for the Low Particle Regime,
2024. DOI: XX.XXXXX/arXiv.XXXX.YYYYY.

"""

import numpy as np
import cbo as opt
import plots as show
from tqdm import trange
import pickle


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
                         Heating=False, Ball=False, Plot=False, Verbose=False)
            Vs_tot[i, j] = min(max(np.log(Vs[-1]), -10), 8)

    show.plot_dependence_on_lambda_and_sigma(Lambdas, Sigmas, Vs_tot, V0)

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
    We try to optimize the Rastriagin function in middle-scale dimensions for
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
                             maxit, 'anisotropic', gamma, dom, function='Rastrigin',
                             Heating=True, Ball=True, Plot=False, Verbose=False)

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
                             Heating=True, Ball=True, Plot=False, Verbose=False)

                Vs_tot[t, :, cnt_d, cnt_gamma] = Vs

    return Vs_tot, dt, maxit, Ds, Gammas
