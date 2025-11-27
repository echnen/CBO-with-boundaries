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
This file contains several useful functions to plot all results of all the
experiments in Sections 5.1 and 5.2 of:

J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
Constrained Consensus-Based Optimization and Numerical Heuristics for the Low
Particle Regime,
2024. DOI: 10.48550/arXiv.2410.10361.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cbo as opt
from scipy.stats import gmean

mpl.rcParams.update({'font.size': 15})


def plot_testing_boundedness_ani(Lambdas, Sigmas, Vs_tot, V0s, title):

    num_of_tests = 2
    fig1, ax = plt.subplots(1, num_of_tests, layout='constrained',
                            figsize=(4 * num_of_tests, 4), sharey=True)

    Titles = ['Constrained', 'Unconstrained']

    for cnt in range(2):

        Vs_M = Vs_tot[cnt]
        V0 = V0s[cnt]

        CS = ax[cnt].contourf(Lambdas, Sigmas, Vs_M.T,
                              levels=np.linspace(-10, 8, 18), cmap='hot')
        ax[cnt].set_xscale('log')
        ax[cnt].set_yscale('log')

        ax[cnt].set_xlabel('Drift parameter' + r' $(\lambda)$')
        if cnt == 0:
            ax[cnt].set_ylabel('Diffusion parameter' + r' $(\sigma)$')

        ax[cnt].contour(CS, levels=[np.log(V0)], colors='lime')

        # ax[cnt].plot(Sigmas ** 2 / 2, Sigmas, color='b')
        ax[cnt].plot(Lambdas, np.sqrt(2 * Lambdas), color='b')
        ax[cnt].set_xlim(Lambdas[0], Lambdas[-1])
        ax[cnt].set_ylim(Sigmas[0], Sigmas[-1])
        ax[cnt].set_title(Titles[cnt])

        if cnt == 0:
            ax[cnt].text(0.5, 1.5, r'$2 \lambda - \sigma^2=0$', color='b',
                         rotation=40)

        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=-10,
                                                                       vmax=8))
    plt.colorbar(sm, ax=ax)
    plt.savefig(title, bbox_inches='tight')
    plt.show()


def plot_testing_boundedness_iso(Lambdas, Sigmas, Vs_tot, V0s, title):

    num_of_tests = 2
    fig1, ax = plt.subplots(1, num_of_tests, layout='constrained',
                            figsize=(4 * num_of_tests, 4), sharey=True)

    Titles = ['Constrained', 'Unconstrained']

    for cnt in range(2):

        Vs_M = Vs_tot[cnt]
        V0 = V0s[cnt]

        CS = ax[cnt].contourf(Lambdas, Sigmas, Vs_M.T,
                              levels=np.linspace(-10, 8, 18), cmap='hot')
        ax[cnt].set_xscale('log')
        ax[cnt].set_yscale('log')

        ax[cnt].set_xlabel('Drift parameter' + r' $(\lambda)$')
        if cnt == 0:
            ax[cnt].set_ylabel('Diffusion parameter ' + r'$(\sigma)$')

        ax[cnt].contour(CS, levels=[np.log(V0)], colors='lime')
        ax[cnt].plot(Lambdas, np.sqrt(Lambdas), color='b')
        ax[cnt].set_xlim(Lambdas[0], Lambdas[-1])
        ax[cnt].set_ylim(Sigmas[0], Sigmas[-1])
        ax[cnt].set_title(Titles[cnt])

        if cnt == 0:
            ax[cnt].text(0.5, 1, r'$\lambda - \sigma^2=0$', color='b',
                         rotation=40)

        sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=-10,
                                                                       vmax=8))
    plt.colorbar(sm, ax=ax)
    plt.savefig(title, bbox_inches='tight')
    plt.show()


def plot_residuals(Vs_tot, dt, title):

    num_of_runs, maxit = Vs_tot.shape

    # creating data
    Times = np.array([k * dt for k in range(maxit)])
    mev = maxit // 10

    plt.figure(figsize=(6, 4))

    for t in range(num_of_runs):

        plt.semilogy(Times, Vs_tot[t, :], color='black',
                     linewidth=1, alpha=0.1)
        plt.xlim(Times[0], Times[-1])

    plt.semilogy(Times, np.mean(Vs_tot, axis=0), '-*', color='black',
                 linewidth=2, markersize=6, markevery=mev)
    plt.xlim(Times[0], Times[-1])
    plt.ylabel(r'$W_2^2(\rho_t^N, \delta_{x^*})$')
    plt.xlabel(r'Time $t$')
    plt.grid(which='both')

    plt.savefig(title, bbox_inches='tight')
    plt.show()


def plot_experiment_large_scale_many(Vs_tot, dt, maxit, Ds, Gammas, title):

    num_of_runs = Vs_tot.shape[0]

    # creating data
    Times = np.array([k * dt for k in range(maxit)])
    mev = maxit // 10

    num_Ds = len(Ds)
    num_Gammas = len(Gammas)
    fig1, ax = plt.subplots(num_Ds, num_Gammas, layout='constrained',
                            figsize=(7, 7), sharey=True, sharex=True)

    for (cnt_d, d) in enumerate(Ds):
        for (cnt_rad, Rad) in enumerate(Gammas):

            for t in range(num_of_runs):

                ax[cnt_d, cnt_rad].semilogy(Times,
                                            Vs_tot[t, :, cnt_d, cnt_rad],
                                            color='black',
                                            linewidth=1, alpha=0.1)
                ax[cnt_d, cnt_rad].set_xlim(Times[0], Times[-1])

            ax[cnt_d, cnt_rad].semilogy(Times,
                                        np.mean(Vs_tot[:, :, cnt_d, cnt_rad],
                                                axis=0), '-*', color='black',
                                        linewidth=2, markersize=6,
                                        markevery=mev, label='mean')
            ax[cnt_d, cnt_rad].set_xlim(Times[0], Times[-1])

            if cnt_rad == 0:
                ax[cnt_d, cnt_rad].set_ylabel(r'$W_2^2(\rho_t^N, \delta_{x^*})$,' + f' for $d={d}$')

            if cnt_d == 0:
                ax[cnt_d, cnt_rad].set_title(rf'$\gamma = {Rad}$')

            if cnt_d == num_Gammas - 1:
                ax[cnt_d, cnt_rad].set_xlabel(r'Time $t$')

            ax[cnt_d, cnt_rad].grid(which='both')

    plt.savefig(title, bbox_inches='tight')
    plt.show()


def plot_domains(hx, Function='Rastrigin', Save=False):

    Save = True
    hx = 0.1

    # plotting experiment 1
    d = 2

    plt.figure(figsize=(10, 8))

    # plotting function
    line_x = line_y = np.arange(-12, 12 + hx, hx)
    Line_x, Line_y = np.meshgrid(line_x, line_y)
    Energy = 10 * 2 + Line_x ** 2 + Line_y ** 2\
        - 10 * np.cos(2 * np.pi * Line_x) - 10 * np.cos(2 * np.pi * Line_y)

    # plotting domain and particles
    Box = np.array([-6.12, 5.12])
    dom = opt.Domain('Box', Box)
    N = 1000
    mean = 5 * np.ones((d, N))
    variance = 1e1
    X = dom.proj(np.random.normal(mean, variance, (d, N)))
    Mask_x = (-6.12 < Line_x) * (Line_x < 5.12)
    Mask_y = (-6.12 < Line_y) * (Line_y < 5.12)
    Mask = Mask_x * Mask_y
    Energy[Mask == False] = np.nan

    plt.contourf(Line_x, Line_y, Energy, cmap='Wistia')
    plt.colorbar()

    plt.scatter(X[0, :], X[1, :], s=10, color='blue', label='Particles')
    plt.scatter(0, 0, s=200, marker='*', color='white')
    plt.scatter(0, 0, s=100, marker='*', label='Global Minimizer',
                color='black')

    plt.legend(bbox_to_anchor=(1.05, -0.05), ncol=2)
    if Save:
        plt.savefig('Results/domain_1.pdf', bbox_inches='tight')
    plt.show()

    # Plotting domain 2
    plt.figure(figsize=(10, 8))

    # plotting function
    line_x = line_y = np.arange(-12, 12 + hx, hx)
    Line_x, Line_y = np.meshgrid(line_x, line_y)
    Energy = 10 * 2 + Line_x ** 2 + Line_y ** 2\
        - 10 * np.cos(2 * np.pi * Line_x) - 10 * np.cos(2 * np.pi * Line_y)

    # plotting domain and particles
    Box = np.array([0, 11.24])
    dom = opt.Domain('Box', Box)
    N = 1000
    mean = 5 * np.ones((d, N))
    variance = 1e1
    X = dom.proj(np.random.normal(mean, variance, (d, N)))
    Mask_x = (-0 < Line_x) * (Line_x < 11.24)
    Mask_y = (-0 < Line_y) * (Line_y < 11.24)
    Mask = Mask_x * Mask_y
    Energy[Mask == False] = np.nan

    plt.contourf(Line_x, Line_y, Energy, cmap='Wistia')
    plt.colorbar()

    plt.scatter(X[0, :], X[1, :], s=10, color='blue', label='Particles')
    plt.scatter(0, 0, s=200, marker='*', color='white')
    plt.scatter(0, 0, s=100, marker='*', label='Global Minimizer',
                color='black')

    plt.legend(bbox_to_anchor=(1.05, -0.05), ncol=2)
    if Save:
        plt.savefig('Results/domain_2.pdf', bbox_inches='tight')
    plt.show()

    # Plotting domain 3
    plt.figure(figsize=(10, 8))

    # plotting function
    line_x = line_y = np.arange(-12, 12 + hx, hx)
    Line_x, Line_y = np.meshgrid(line_x, line_y)
    Energy = 10 * 2 + Line_x ** 2 + Line_y ** 2\
        - 10 * np.cos(2 * np.pi * Line_x) - 10 * np.cos(2 * np.pi * Line_y)

    # plotting domain and particles
    dom = opt.Domain('noncvx2', [np.zeros(2), 5.12, 2])
    N = 1000
    mean = 5 * np.ones((d, N))
    variance = 1e1
    X = dom.proj(np.random.normal(mean, variance, (d, N)))
    Mask_1 = Line_x ** 2 + Line_y ** 2 < 5.12 ** 2
    Mask_2 = (Line_x - 1) ** 2 + Line_y ** 2 > 1
    Mask_3 = (Line_x + 1) ** 2 + (Line_y) ** 2 > 1
    Mask = Mask_1 * Mask_2 * Mask_3
    Energy[Mask == False] = np.nan

    plt.contourf(Line_x, Line_y, Energy, cmap='Wistia')
    plt.colorbar()

    plt.scatter(X[0, :], X[1, :], s=10, color='blue', label='Particles')
    plt.scatter(0, 0, s=200, marker='*', color='white')
    plt.scatter(0, 0, s=100, marker='*', label='Global Minimizer',
                color='black')

    plt.legend(bbox_to_anchor=(1.05, -0.05), ncol=2)
    if Save:
        plt.savefig('Results/domain_3.pdf', bbox_inches='tight')
    plt.show()

    # Plotting domain 3
    plt.figure(figsize=(10, 8))

    # plotting function
    line_x = line_y = np.arange(-12, 12 + hx, hx)
    Line_x, Line_y = np.meshgrid(line_x, line_y)
    Energy = 10 * 2 + Line_x ** 2 + Line_y ** 2\
        - 10 * np.cos(2 * np.pi * Line_x) - 10 * np.cos(2 * np.pi * Line_y)

    # plotting domain and particles
    dom = opt.Domain('noncvx1', [np.zeros(2), 5.12, 2])
    N = 1000
    mean = 5 * np.ones((d, N))
    variance = 1e1
    X = dom.proj(np.random.normal(mean, variance, (d, N)))
    Mask_1 = Line_x ** 2 + Line_y ** 2 < 5.12 ** 2
    Mask_2 = (Line_x - 1) ** 2 + (Line_y - 1) ** 2 > 2
    Mask_3 = (Line_x + 1) ** 2 + (Line_y + 1) ** 2 > 2
    Mask = Mask_1 * Mask_2 * Mask_3
    Energy[Mask == False] = np.nan

    plt.contourf(Line_x, Line_y, Energy, cmap='Wistia')
    plt.colorbar()

    plt.scatter(X[0, :], X[1, :], s=10, color='blue', label='Particles')
    plt.scatter(0, 0, s=200, marker='*', color='white')
    plt.scatter(0, 0, s=100, marker='*', label='Global Minimizer', color='black')

    plt.legend(bbox_to_anchor=(1.05, -0.05), ncol=2)
    if Save:
        plt.savefig('Results/domain_4.pdf', bbox_inches='tight')
    plt.show()


def plot_experiment_heuristics(Vs_tot, dt, maxit, Ss):
    """
    Plotting experiment heurisitcs
    """

    num_of_runs, maxit, ref_S, _ = Vs_tot.shape

    # the case with only Heuristic Exploration activated
    plt.figure(figsize=(6, 4))
    for t in range(num_of_runs):

        plt.semilogy(Ss, Vs_tot[t, -1, :, 0], '*', color='black',
                     linewidth=1, alpha=0.1)

    plt.semilogy(Ss, gmean(Vs_tot[:, -1, :, 0], axis=0), color='black',
                 linewidth=2, alpha=1)
    plt.xlim(Ss[0], Ss[-1])
    plt.ylabel(r'$W_2^2(\rho_T^N, \delta_{x^*})$')
    plt.xlabel(r'Exploration Parameter $S$')
    plt.ylim(1e-13, 1e1)
    plt.xlim(Ss[0], 8)
    plt.axvline(x=1, color='red', linewidth=2)
    plt.grid(which='both')

    plt.savefig("Results/testing_heuristics_with_ball.pdf", bbox_inches='tight')
    plt.show()

    # the case with Heuristic Ball activated
    plt.figure(figsize=(6, 4))
    for t in range(num_of_runs):

        plt.semilogy(Ss, Vs_tot[t, -1, :, 1], '*', color='black',
                     linewidth=1, alpha=0.1)

    plt.semilogy(Ss, gmean(Vs_tot[:, -1, :, 1], axis=0), color='black',
                 linewidth=2, alpha=1)
    plt.xlim(Ss[0], Ss[-1])
    plt.ylim(1e-13, 1e1)
    plt.xlim(Ss[0], 8)
    plt.axvline(x=1, color='red', linewidth=2)
    plt.ylabel(r'$W_2^2(\rho_T^N, \delta_{x^*})$')
    plt.xlabel(r'Exploration Parameter $S$')
    plt.grid(which='both')

    plt.savefig("Results/testing_heuristics_without_ball.pdf", bbox_inches='tight')
    plt.show()
