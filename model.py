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


import sys
import os
import pickle

import numpy as np
import pyfftw as fftw
import matplotlib.pyplot as plt
import matplotlib.colors as mcl


# Define blue colormap with nice blue.
cdict = {
    'red': ((0.0, 0.0, 0.0), (0.5, 0.216, 0.216), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0), (0.5, 0.494, 0.494), (1.0, 1.0, 1.0)),
    'blue': ((0.0, 0.0, 0.0), (0.5, 0.722, 0.722), (1.0, 1.0, 1.0))}
cm_blues = mcl.LinearSegmentedColormap('cm_blues', cdict, 256)


class Model:

    """
    General QG model

    This is the skeleton of a QG model that consists of a number of conserved
    quantities that are advected horizontally.  Implementations of this model
    need to specify the number of conserved quantities (nz) and supply an
    inversion relation that yield the streamfuncion given the conserved quan-
    tities.  The model geometry is doubly periodic in the perturbations; mean
    flow and gradients in the conserved quantities can be prescribed.
    """

    def __init__(self, nz):
        self.nz = nz            # number of levels
        self.u = np.zeros(nz)   # mean zonal velocities
        self.v = np.zeros(nz)   # mean meridional velocities
        self.qx = np.zeros(nz)  # mean zonal PV gradient
        self.qy = np.zeros(nz)  # mean meridional PV gradient

    def linstab(self, k, l):
        """
        Perform linear stability analysis for wavenumbers k and l.
        
        Returns complex eigen-frequenies omega of the eigenvalue problem
          [k U + l V + (k Qy - l Qx) L^-1] q = omega q.
        The phase speeds are Re omega/k and the growth rates Im omega.
        """
        # Set up inversion matrix with specified wavenumbers.
        L = self.invmatrix(k[np.newaxis,:,np.newaxis],
            l[:,np.newaxis,np.newaxis])
        # Allow proper broadcasting over matrices.
        kk = k[np.newaxis,:,np.newaxis,np.newaxis]
        ll = l[:,np.newaxis,np.newaxis,np.newaxis]
        # Compute (k Gy - l Gx) L^-1.
        GL = np.einsum('...ij,...jk->...ik', kk*np.diag(self.qy)
            - ll*np.diag(self.qx), np.linalg.inv(L))
        # Solve eigenvalue problem.
        w, v = np.linalg.eig(kk*np.diag(self.u) + ll*np.diag(self.v) + GL)
        # Sort eigenvalues.
        w.sort()
        return w

    def initnum(self, a, n, dt):
        """Initialize numerics."""
        self.a = a          # domain size
        self.n = n          # number of Fourier modes per direction
        self.dt = dt        # time step
        self.diffexp = 2    # exponent of diffusion operator
        self.hypodiff = 0.  # hypodiffusion coefficient
        self.threads = 1    # number of threads for FFT
        self.time = 0.      # initial simulation time
        # Set up grid.
        self.grid()
        # Set up inversion matrix.
        self.L = self.invmatrix(self.k, self.l)

    def grid(self):
        """Set up spectral and physical grid."""
        # Set up spectral grid.
        k = abs(np.fft.fftfreq(self.n, d=self.a/(2*np.pi*self.n))[:self.n/2+1])
        l = np.fft.fftfreq(self.n, d=self.a/(2*np.pi*self.n))
        self.k = k[np.newaxis,:,np.newaxis]
        self.l = l[:,np.newaxis,np.newaxis]
        # Set up physical grid.
        x = np.arange(self.n) * self.a / self.n
        y = np.arange(self.n) * self.a / self.n
        self.x = x[np.newaxis,:,np.newaxis]
        self.y = y[:,np.newaxis,np.newaxis]

    def initq(self, qp):
        """Transform qp to spectral space and initialize q."""
        self.q = fftw.interfaces.numpy_fft.rfft2(qp, axes=(0, 1),
            threads=self.threads)
        self.q[:,0,0] = 0.  # ensuring zero mean

    def timestep(self):
        """Perform time step."""
        self.advection()
        self.diffusion()
        self.time += self.dt

    def advection(self):
        """Perform RK4 step for advective terms (linear and nonlinear)."""
        q1 = self.advrhs(self.q)
        q2 = self.advrhs(self.q + self.dt*q1/2)
        q3 = self.advrhs(self.q + self.dt*q2/2)
        q4 = self.advrhs(self.q + self.dt*q3)
        self.q += self.dt*(q1 + 2*q2 + 2*q3 + q4)/6

    def diffusion(self):
        """Perform implicit (hyper- and hypo-) diffusion step."""
        k2 = self.k**2 + self.l**2
        k2[0,0,:] = 1.  # preventing div. by zero for wavenumber 0
        self.q *= np.exp(-self.nu * k2**(self.diffexp/2.) * self.dt)
        if self.hypodiff > 0:
            self.q *= np.exp(-self.hypodiff / k2 * self.dt)

    def advrhs(self, q):
        """
        Calculate advective terms on RHS of PV equation.
        
        Calculate mean-eddy and eddy-eddy advection terms:
            u q'_x + v q'_y + u' q_x + v' q_y + J(p', q')
        """
        # Perform inversion.
        p = np.linalg.solve(self.L, q)
        # Calculate RHS.
        rhs = \
            - 1j * (self.k*self.u + self.l*self.v) * q \
            - 1j * (self.k*self.qy - self.l*self.qx) * p \
            - self.jacobian(p, q)
        return rhs

    def jacobian(self, A, B):
        """
        Calculate Jacobian A_x B_y - A_y B_x.
        
        Transform Ax, Ay, Bx, By to physical space, perform multi-
        plication and subtraction, and transform back to spectral space.
        To avoid aliasing, apply 3/2 padding.
        """
        Axp = self.ifft_pad(1j * self.k * A)
        Ayp = self.ifft_pad(1j * self.l * A)
        Bxp = self.ifft_pad(1j * self.k * B)
        Byp = self.ifft_pad(1j * self.l * B)
        return self.fft_truncate(Axp * Byp - Ayp * Bxp)

    def fft_truncate(self, up):
        """Perform forward FFT on physical field up and truncate (3/2 rule)."""
        us = fftw.interfaces.numpy_fft.rfft2(up, axes=(0, 1),
            threads=self.threads)
        u = np.zeros((self.n, self.n/2 + 1, self.nz), dtype=complex)
        u[: self.n/2, :, :] = us[: self.n/2, : self.n/2 + 1, :]
        u[self.n/2 :, :, :] = us[self.n : 3*self.n/2, : self.n/2 + 1, :]
        return u/2.25  # accounting for normalization

    def ifft_pad(self, u):
        """Pad spectral field u (3/2 rule) and perform inverse FFT."""
        us = np.zeros((3*self.n/2, 3*self.n/4 + 1, self.nz), dtype=complex)
        us[: self.n/2, : self.n/2 + 1, :] = u[: self.n/2, :, :]
        us[self.n : 3*self.n/2, : self.n/2 + 1, :] = u[self.n/2 :, :, :]
        return fftw.interfaces.numpy_fft.irfft2(2.25*us, axes=(0, 1),
            threads=self.threads)

    def doubleres(self):
        """Double the resolution, interpolate fields."""
        self.n *= 2
        # Pad spectral field.
        qs = np.zeros((self.n, self.n/2 + 1, self.nz), dtype=complex)
        qs[: self.n/4, : self.n/4 + 1, :] = self.q[: self.n/4, :, :]
        qs[3*self.n/4 : self.n, : self.n/4 + 1, :] = self.q[self.n/4 :, :, :]
        # Account for normalization.
        self.q = 4*qs
        # Update grid.
        self.grid()
        # Update inversion matrix.
        self.L = self.invmatrix(self.k, self.l)

    def screenlog(self):
        """Print model state info on screen."""
        # Write time (in seconds).
        sys.stdout.write(' {:15.0f}'.format(self.time))
        # Write mean enstrophy for each layer.
        for i in range(self.nz):
            sys.stdout.write(' {:5e}'.format(np.mean(np.abs(self.q[:,:,i])**2)
                /self.n**2))
        sys.stdout.write('\n')

    def snapshot(self, name):
        """Save snapshots of total q (mean added) for each level."""
        # Check whether directory exists.
        if not os.path.isdir(name + '/snapshots'):
            os.makedirs(name + '/snapshots')
        # Transform to physical space.
        qp = fftw.interfaces.numpy_fft.irfft2(self.q, axes=(0, 1),
            threads=self.threads)
        # Add mean gradients
        qp += self.qx * (self.x - self.a / 2)
        qp += self.qy * (self.y - self.a / 2)
        # Determine range of colorbars.
        m = np.max([np.abs(self.qx * self.a), np.abs(self.qy * self.a)], axis=0)
        # Save image for each layer.
        for i in range(self.nz):
            plt.imsave(
                name + '/snapshots/{:03d}_{:015.0f}.png'.format(i, self.time),
                qp[:,:,i], origin='lower', vmin=-m[i], vmax=m[i], cmap=cm_blues)

    def save(self, name):
        """Save model state."""
        # Check whether directory exists.
        if not os.path.isdir(name + '/data'):
            os.makedirs(name + '/data')
        # Save.
        f = open(name + '/data/{:015.0f}'.format(self.time), 'w')
        pickle.dump(self, f)


def load(name, time):
    """Load model state."""
    f = open(name + '/data/{:015.0f}'.format(time), 'r')
    return pickle.load(f)


class TwoDim(Model):

    """
    Two dimensional model

    This implements a two-dimensional model that consists of vorticity conser-
    vation.  The inversion relation is simply a Poisson equation.
    """

    def __init__(self):
        Model.__init__(self, 1)

    def initmean(self, qx, qy):
        """Set up the mean state."""
        # Set up PV gradients.
        self.qx[0] = qx
        self.qy[0] = qy

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        k2 = (k**2 + l**2)[:,:,0]
        k2[k2 == 0.] = 1.  # preventing div. by zero for wavenumber 0
        L = np.empty((l.size, k.size, 1, 1))
        L[:,:,0,0] = - k2
        return L


class Layered(Model):

    """
    Multi-layer model

    This implements a multi-layer model with a rigid lid.  See Vallis (2006)
    for the formulation and notation.  The layer thicknisses h, buoyancy
    jumps g, mean flows (u, v), and beta can be passed to initmean.
    """

    def initmean(self, f, h, g, u, v, beta):
        """Set up the mean state."""
        self.f = np.array(f)  # Coriolis parameter
        self.h = np.array(h)  # layer thicknesses
        self.g = np.array(g)  # buoyancy jumps at interfaces
        self.u = np.array(u)  # mean zonal flow
        self.v = np.array(v)  # mean meridional flow
        # Set up mean PV gradients.
        self.qx = np.zeros(self.nz)
        self.qx[:-1] -= f**2 * (self.v[:-1] - self.v[1:]) / (self.h[:-1] * g)
        self.qx[1:] += f**2 * (self.v[:-1] - self.v[1:]) / (self.h[1::] * g)
        self.qy = beta * np.ones(self.nz)
        self.qy[:-1] += f**2 * (self.u[:-1] - self.u[1:]) / (self.h[:-1] * g)
        self.qy[1:] -= f**2 * (self.u[:-1] - self.u[1:]) / (self.h[1::] * g)

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        k2 = (k**2 + l**2)[:,:,0]
        k2[k2 == 0.] = 1.  # preventing div. by zero for wavenumber 0
        L = np.zeros((l.size, k.size, self.nz, self.nz))
        for i in range(self.nz):
            L[:,:,i,i] -= k2
            if i > 0:
                L[:,:,i,i-1] += self.f**2 / (self.h[i] * self.g[i-1])
                L[:,:,i,i] -= self.f**2 / (self.h[i] * self.g[i-1])
            if i < self.nz - 1:
                L[:,:,i,i] -= self.f**2 / (self.h[i] * self.g[i])
                L[:,:,i,i+1] += self.f**2 / (self.h[i] * self.g[i])
        return L


class Eady(Model):

    """
    Eady model

    This implements an Eady model that consists of surface and bottom buoyancy
    conservation and implicit interior dynamics determined by zero PV there.
    The conserved quantities here are PV-like:
      q[0] = - f b(0) / N^2,
      q[1] = + f b(-H) / N^2.
    """

    def __init__(self):
        Model.__init__(self, 2)

    def initmean(self, f, N, H, Sx, Sy):
        """Set up the mean state."""
        self.f = f  # Coriolis parameter
        self.N = N  # buoyancy frequency
        self.H = H  # depth
        # Set up mean flow.
        self.u = np.array([0, - Sx * H])
        self.v = np.array([0, - Sy * H])
        # Set up mean PV gradients.
        self.qx = np.array([- f**2 * Sy / N**2, + f**2 * Sy / N**2])
        self.qy = np.array([+ f**2 * Sx / N**2, - f**2 * Sx / N**2])

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        kh = np.hypot(k, l)[:,:,0]
        kh[kh == 0.] = 1.  # preventing div. by zero for wavenumber 0
        mu = self.N * kh * self.H / self.f
        L = np.empty((l.size, k.size, 2, 2))
        L[:,:,0,0] = - self.f * kh / (self.N * np.tanh(mu))
        L[:,:,0,1] = + self.f * kh / (self.N * np.sinh(mu))
        L[:,:,1,0] = + self.f * kh / (self.N * np.sinh(mu))
        L[:,:,1,1] = - self.f * kh / (self.N * np.tanh(mu))
        return L


class FloatingEady(Model):

    """
    Floating Eady model

    This implements a "floating" Eady model that consists of a layer of
    constant PV coupled to an infinitely deep layer below that also has
    constant PV (see Callies, Flierl, Ferrari, Fox-Kemper, 2015).  The model
    consists of two conserved PV-like quantities at the surface and the inter-
    face between the layers:
      q[0] = - f b(0) / N[0]^2,
      q[1] = + f [b^+(-H) / N[0]^2 - b^-(-H) / N[1]^2,
    where N[0] and N[1] are the buoyancy frequencies of the Eady and deep
    layers, respectively.
    """

    def __init__(self):
        Model.__init__(self, 2)

    def initmean(self, f, N, H, Sx, Sy):
        """Set up the mean state."""
        self.f = f            # Coriolis parameter
        self.N = np.array(N)  # buoyancy frequencies of the two layers
        self.H = H            # depth of upper layer
        # Set up mean flow.
        self.u = np.array([0, - Sx[0] * H])
        self.v = np.array([0, - Sy[0] * H])
        # Set up mean PV gradients.
        self.qx = np.array([
            - f**2 * Sy[0] / N[0]**2,
            + f**2 * (Sy[0] / N[0]**2 - Sy[1] / N[1]**2)])
        self.qy = np.array([
            + f**2 * Sx[0] / N[0]**2,
            - f**2 * (Sx[0] / N[0]**2 - Sx[1] / N[1]**2)])

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        kh = np.hypot(k, l)[:,:,0]
        kh[kh == 0.] = 1.  # preventing div. by zero for wavenumber 0
        mu = self.N[0] * kh * self.H / self.f
        L = np.empty((l.size, k.size, 2, 2))
        L[:,:,0,0] = - self.f * kh / (self.N[0] * np.tanh(mu))
        L[:,:,0,1] = + self.f * kh / (self.N[0] * np.sinh(mu))
        L[:,:,1,0] = + self.f * kh / (self.N[0] * np.sinh(mu))
        L[:,:,1,1] = - self.f * kh / (self.N[0] * np.tanh(mu)) \
            - self.f * kh / self.N[1]
        return L


class TwoEady(Model):

    """
    Two-Eady model

    This implements a two-Eady model that consists of two layers of
    constant PV coupled at a deformable interface (see Callies, Flierl,
    Ferrari, Fox-Kemper, 2015).  The model consists of three conserved PV-like
    quantities at the surface, the interface between the layers, and the
    bottom:
      q[0] = - f b(0) / N[0]^2,
      q[1] = + f [b^+(-H[0]) / N[0]^2 - b^-(-H[0]) / N[1]^2,
      q[0] = + f b(-H[0]-H[1]) / N[1]^2,
    where N[0] and N[1] are the buoyancy frequencies of the two layers and H[0]
    and H[1] are their depths.
    """

    def __init__(self):
        Model.__init__(self, 3)

    def initmean(self, f, N, H, Sx, Sy):
        """Set up the mean state."""
        self.f = f            # Coriolis parameter
        self.N = np.array(N)  # buoyancy frequencies of the two layers
        self.H = np.array(H)  # depths of the two layers
        # Set up mean flow.
        self.u = np.array([0, - Sx[0] * H[0], - Sx[0] * H[0] - Sx[1] * H[1]])
        self.v = np.array([0, - Sy[0] * H[0], - Sy[0] * H[0] - Sy[1] * H[1]])
        # Set up mean PV gradients.
        self.qx = np.array([
            - f**2 * Sy[0] / N[0]**2,
            + f**2 * (Sy[0] / N[0]**2 - Sy[1] / N[1]**2),
            + f**2 * Sy[1] / N[1]**2])
        self.qy = np.array([
            + f**2 * Sx[0] / N[0]**2,
            - f**2 * (Sx[0] / N[0]**2 - Sx[1] / N[1]**2),
            - f**2 * Sx[1] / N[1]**2])

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        kh = np.hypot(k, l)
        kh[kh == 0.] = 1.  # preventing div. by zero for wavenumber 0
        mu = self.N * kh * self.H / self.f
        kh = kh[:,:,0]
        L = np.zeros((l.size, k.size, 3, 3))
        L[:,:,0,0] = - self.f * kh / (self.N[0] * np.tanh(mu[:,:,0]))
        L[:,:,0,1] = + self.f * kh / (self.N[0] * np.sinh(mu[:,:,0]))
        L[:,:,1,0] = + self.f * kh / (self.N[0] * np.sinh(mu[:,:,0]))
        L[:,:,1,1] = - self.f * kh / (self.N[0] * np.tanh(mu[:,:,0])) \
            - self.f * kh / (self.N[1] * np.tanh(mu[:,:,1]))
        L[:,:,1,2] = + self.f * kh / (self.N[1] * np.sinh(mu[:,:,1]))
        L[:,:,2,1] = + self.f * kh / (self.N[1] * np.sinh(mu[:,:,1]))
        L[:,:,2,2] = - self.f * kh / (self.N[1] * np.tanh(mu[:,:,1]))
        return L


class TwoEadyJump(Model):

    """
    Two-Eady model with jump

    This is a two-Eady model with an additional buoyancy and velocity jump
    at the interface.  The model then has an additional conserved quantity,
    as if there was an additional layer at the interface.  (The jump can be
    thought of as an additional, infinitesimally thin layer with infinite
    stratification and shear.)  The conserved quantities are now
      q[0] = - f b(0) / N[0]^2,
      q[1] = + f b^+(-H[0]) / N[0]^2 + f \eta,
      q[1] = - f b^-(-H[0]) / N[1]^2 - f \eta,
      q[0] = + f b(-H[0]-H[1]) / N[1]^2,
    where \eta is the interface displacement.  The buoyancy jump g' and the
    velocity jump (U, V) can be passed to initmean.  
    """

    def __init__(self):
        Model.__init__(self, 4)

    def initmean(self, f, N, H, g, Sx, Sy, U, V):
        """Set up the mean state."""
        self.f = f            # Coriolis parameter
        self.N = np.array(N)  # buoyancy frequencies of the two layers
        self.H = np.array(H)  # depths of the two layers
        self.g = g            # buoyancy jump at interface
        # Set up mean flow.
        self.u = np.array([0, - Sx[0] * H[0], - Sx[0] * H[0] - U,
            - Sx[0] * H[0] - U - Sx[1] * H[1]])
        self.v = np.array([0, - Sy[0] * H[0], - Sy[0] * H[0] - V,
            - Sy[0] * H[0] - V - Sy[1] * H[1]])
        # Set up mean PV gradients.
        self.qx = np.array([
            - f**2 * Sy[0] / N[0]**2,
            + f**2 * Sy[0] / N[0]**2 - f**2 * V / g,
            - f**2 * Sy[1] / N[1]**2 + f**2 * V / g,
            + f**2 * Sy[1] / N[1]**2])
        self.qy = np.array([
            + f**2 * Sx[0] / N[0]**2,
            - f**2 * Sx[0] / N[0]**2 + f**2 * U / g,
            + f**2 * Sx[1] / N[1]**2 - f**2 * U / g,
            - f**2 * Sx[1] / N[1]**2])

    def invmatrix(self, k, l):
        """Set up the inversion matrix L."""
        kh = np.hypot(k, l)
        kh[kh == 0.] = 1.  # preventing div. by zero for wavenumber 0
        mu = self.N * kh * self.H / self.f
        kh = kh[:,:,0]
        L = np.zeros((l.size, k.size, 4, 4))
        L[:,:,0,0] = - self.f * kh / (self.N[0] * np.tanh(mu[:,:,0]))
        L[:,:,0,1] = + self.f * kh / (self.N[0] * np.sinh(mu[:,:,0]))
        L[:,:,1,0] = + self.f * kh / (self.N[0] * np.sinh(mu[:,:,0]))
        L[:,:,1,1] = - self.f * kh / (self.N[0] * np.tanh(mu[:,:,0])) \
            - self.f**2 / self.g
        L[:,:,1,2] = self.f**2 / self.g
        L[:,:,2,1] = self.f**2 / self.g
        L[:,:,2,2] = - self.f * kh / (self.N[1] * np.tanh(mu[:,:,1])) \
            - self.f**2 / self.g
        L[:,:,2,3] = + self.f * kh / (self.N[1] * np.sinh(mu[:,:,1]))
        L[:,:,3,2] = + self.f * kh / (self.N[1] * np.sinh(mu[:,:,1]))
        L[:,:,3,3] = - self.f * kh / (self.N[1] * np.tanh(mu[:,:,1]))
        return L
