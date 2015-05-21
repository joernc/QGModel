# Copyright (C) 2013,2014,2015 Joern Callies
#
# This file is part of QGModel.
#
# QGModel is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QGModel is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with QGModel.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np

import model


# Initialization: initialize model of required type, set up mean state,
# set up initial state, and save initial snapshot.  Given here are three
# examples, all taken from Callies, Flierl, Ferrari, Fox-Kemper (2015).
# These are low-resolution versions of the simulations in the paper.
# For the second and third case, the resolution is insufficient to re-
# solve the instabilities properly.  These low-resolution models can be
# used to spin up higher-resolution versions (see below).  The time step
# chosen here may be too large for the fully turbulent regime.  The
# model can be restarted with a reduced time step when it blows up (see
# below).

folder = 'eady'
m = model.Eady(5e5, 128, 5000.)
m.setup(1e-4, 8e-3, 500., 1e-4, 0.)
m.initq(1e-5 * np.random.rand(128, 128, 2))
m.snapshot(folder)

#folder = 'fleady'
#m = model.FloatingEady(5e5, 128, 5000.)
#m.setup(1e-4, [2e-3, 8e-3], 100., [1e-4, 1e-4], [0., 0.])
#m.initq(1e-7 * np.random.rand(128, 128, 2))
#m.snapshot(folder)

#folder = 'two-eady'
#m = model.TwoEady(5e5, 128, 5000.)
#m.setup(1e-4, [2e-3, 8e-3], [100., 400.], [1e-4, 1e-4], [0., 0.])
#m.initq(1e-7 * np.random.rand(128, 128, 3))
#m.snapshot(folder)

# More setup: Here we define the hyper- and hypoviscosity.  In this
# case, high-order hyperviscosity is used, with a coefficient that is
# adopted from Callies et al. (2015) to account for the the reduced
# resolution.

m.nu = 2.5e46 * 4.**20
m.diffexp = 20
m.hypodiff = 1e-16

# Load model state: This simple command allows one to restart the model
# from a previously saved state.  All that is necessary is the folder
# and the time from which the model is to be restarted.

#m = model.load(folder, 50000000)

# Increase resolution: This increases the model resolution and interpo-
# lates the current state onto the finer grid.  We also reduce the
# hyperviscosity coefficient to allow more small-scale structure.

#m.doubleres()
#m.nu /= 2.**20

# Reduce time step: This cuts the time step in half, which may be
# necessary as the model becomes more energetic or the resolution is
# increased.

#m.dt /= 2.

# Step model forward: This is the main loop of the model.  It steps
# forward the model, prints information on the model state, and episo-
# dically saves the model state to file and a snapshot as a png-file.

for i in range(20000):
    m.timestep()
    m.screenlog()
    if np.mod(m.time, 250000) == 0:
        m.save(folder)
        m.snapshot(folder)
