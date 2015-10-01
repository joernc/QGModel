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
import matplotlib.pyplot as plt

import model


# Initialization: initialize model of required type and set up mean
# state.  Given here are five examples, the latter three described in
# Callies, Flierl, Ferrari, Fox-Kemper (2015).  These are low-
# resolution versions of the simulations in the paper.  For the floating
# Eady and two-Eady case, the resolution is insufficient to resolve the
# instabilities properly.  These low-resolution models can be used to
# spin up higher-resolution versions (see below).  The time step chosen
# here may be too large for the fully turbulent regime.  The model can
# be restarted with a reduced time step when it blows up (see below).

#folder = 'two-dim'
#m = model.TwoDim()
#m.initmean(0, 2e-11)

#folder = 'sqg'
#m = model.Surface()
#m.initmean(1e-4, 8e-4, 1e-4, 0)

#folder = 'two-layer'
#m = model.Layered(2)
#m.initmean(1e-4, 2*[250.], [1.6e-2], [2.5e-2, 0.], 2*[0.], 0.)

folder = 'eady'
m = model.Eady()
m.initmean(1e-4, 8e-3, 500., 1e-4, 0.)

#folder = 'fleady'
#m = model.FloatingEady()
#m.initmean(1e-4, [2e-3, 8e-3], 100., [1e-4, 1e-4], [0., 0.])

#folder = 'two-eady'
#m = model.TwoEady()
#m.initmean(1e-4, [2e-3, 8e-3], [100., 400.], [1e-4, 1e-4], [0., 0.])

#folder = 'two-eady-jump'
#m = model.TwoEadyJump()
#m.initmean(1e-4, [2e-3, 8e-3], [100., 400.], 4e-3, [1e-4, 1e-4], [0., 0.],
#    2e-2, 0)

# Perform linear stability analysis.  This is an example of how to
# calculate the linear phase speeds and growth rates for a set of
# specified wavenumbers k and l for the model set up above.

k = 2 * np.pi * np.logspace(-6, -3, 500)
l = np.zeros(1)
w = m.linstab(k, l)

fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].semilogx(k / (2*np.pi), np.real(w[0,:,:]) / k[:,np.newaxis])
ax[1].semilogx(k / (2*np.pi), np.imag(w[0,:,:]))
ax[0].set_ylabel('phase speed')
ax[1].set_ylabel('growth rate')
ax[1].set_xlabel('inverse wavelength')
plt.show()

# Set up the numerics: Here we initialize the numerics, specifying
# domain size, resolution and time step.  Note that initnum must always
# be called after initmean has been used to set up the mean state and
# model parameters, because these are used in setting up the inversion.
# We also define the hyper- and hypoviscosity.  In this case, high-order
# hyperviscosity is used, with a coefficient that is modified from
# Callies et al. (2015) to account for the the reduced resolution.

m.initnum(5e5, 128, 5000.)
m.initq(1e-4 * np.random.rand(128, 128, m.nz))
m.snapshot(folder)

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
    if m.clock % 250000. == 0:
        m.save(folder)
        m.snapshot(folder)
