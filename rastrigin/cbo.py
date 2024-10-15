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
This file contains an implementation of a possibly constrained CBO scheme
with heuristics detailed in:

J. Beddrich, E. Chenchene, M. Fornasier, H. Huang, B. Wohlmuth.
Constrained Consensus-Based Optimization and Numerical Heuristics for the Low
Particle Regime,
2024. DOI: 10.48550/arXiv.2410.10361.

"""

import numpy as np


class Domain:
    def __init__(self, dom_type, sizes=None):

        self.sizes = sizes
        self.dom_type = dom_type

    def proj_ball(self, X, center, radius):

        X_out = X - center[:, np.newaxis]
        Norms = np.linalg.norm(X_out, axis=0)
        X_out[:, Norms > radius] = radius * X_out[:, Norms > radius] / \
            Norms[Norms > radius]
        X_out += center[:, np.newaxis]

        return X_out

    def proj_coball(self, X, center, radius):

        X_out = X - center[:, np.newaxis]
        Norms = np.linalg.norm(X_out, axis=0)
        X_out[:, Norms < radius] = radius * X_out[:, Norms < radius] / \
            Norms[Norms < radius]
        X_out += center[:, np.newaxis]

        return X_out

    def proj(self, X):

        if self.dom_type == 'Box':

            return np.minimum(np.maximum(X, self.sizes[0]), self.sizes[1])

        elif self.dom_type == 'Ball':

            return self.proj_ball(X, self.sizes[0], self.sizes[1])

        elif self.dom_type == 'noncvx1':

            # first, we project onto larger domain
            d = self.sizes[2]
            X = self.proj_ball(X, self.sizes[0], self.sizes[1])

            # projection onto actual domain
            small_balls = [(1, np.array([1] + [0] * (d - 1))),
                           (1, -np.array([1] + [0] * (d - 1)))]

            for radius_small, center_small in small_balls:
                X = self.proj_coball(X, center_small, radius_small)

            return X

        elif self.dom_type == 'noncvx2':

            # first, we project onto larger domain
            d = self.sizes[2]
            X = self.proj_ball(X, self.sizes[0], self.sizes[1])

            # projection onto actual domain
            small_balls = [(1, -np.ones(d) / np.sqrt(d)),
                           (1, np.ones(d) / np.sqrt(d))]

            for radius_small, center_small in small_balls:
                X = self.proj_coball(X, center_small, radius_small)

            return X

        elif self.dom_type == 'unconstrained':
            return X

        else:
            raise AttributeError("Please select an available domain!")


def compute_consensus(alpha, X, d, N, function='Rastrigin'):

    if function == 'Rastrigin':
        E = 10 * d + np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X), axis=0)

    else:
        E = np.zeros((d, N))

    E_min = min(E)
    W = np.exp(- alpha * (E - E_min))
    tot_mass = np.sum(W)

    return np.sum(W[np.newaxis, :] * X, axis=1) / tot_mass


def heuristics_ball(X, x_alpha, Rad):

    X_proj = X.copy() - x_alpha[:, np.newaxis]
    d = X.shape[0]
    norms = np.sqrt(np.sum(X_proj ** 2, axis=0))
    max_norms = np.max(norms)
    where = norms > Rad * max_norms

    # projecting X onto ball centered in x_alpha with radius Rad * max_norms
    X_proj[:, where] = Rad * max_norms * X_proj[:, where] / \
        np.repeat(norms[np.newaxis, where], d, axis=0)
    X_proj = x_alpha[:, np.newaxis] + X_proj

    return X_proj


def cbo(dt, sig, lam, alpha, X0, d, N, x_opt, maxit,
        case, gamma, dom, function='Rastrigin',
        Heating=False, Ball=False, Plot=False, Verbose=False, Project=True):

    # setting parameters
    sdt = np.sqrt(dt)

    # initialization
    X = np.copy(X0)

    # useful objects
    Vs = np.zeros(maxit)

    # initial iteration
    it = 0

    # step for heating (we increase up to 1e9 linearly)
    if Heating:
        step_alpha = (1e9 - alpha) / maxit

    # optional useful parameters (only for plotting)
    show_every = maxit // 5

    while it < maxit:

        # compute current consensus point
        x_alpha = compute_consensus(alpha, X, d, N, function)

        # compute Wasserstein distance to minimizer
        V_it = np.sum((X - x_opt[:, np.newaxis]) ** 2) / N
        Vs[it] = V_it

        # Brownian motion for exploration term
        dB = np.random.normal(0, sdt, (d, N))

        # particle update step (according to SDE)
        if case == 'anisotropic':
            X = dom.proj(X - lam * (X - x_alpha[:, np.newaxis]) * dt +
                         sig * np.abs(X - x_alpha[:, np.newaxis]) * dB)

        if case == 'isotropic':
            X = dom.proj(X - lam * (X - x_alpha[:, np.newaxis]) * dt +
                         sig * (np.linalg.norm(X - x_alpha[:, np.newaxis],
                                               axis=0)[np.newaxis, :]) * dB)

        # optimistic drift
        if Ball:
            X = heuristics_ball(X, x_alpha, gamma)

        # increasing alpha
        if Heating:
            alpha += step_alpha

        # showing status
        if it % show_every == 0 and it > show_every and Verbose:
            print('|| Iteration: {:<5} | Residual: {:<25} ||'.format(it, V_it))

        it += 1

    return Vs
