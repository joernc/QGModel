# QGModel
This is Python code that solves the nonlinear QG equations for PV anomalies on a doubly-periodic domain.  It consists of two parts:

1. a fully-dealiased pseudo-spectral code that steps forward the PV conservation equations and
2. implementations of specific model types that provide an inversion relation.

At the moment, six model types have been implemented:

1. two-dimensional dynamics (`TwoDim`),
2. surface QG dynamics (`Surface`),
3. multi-layer dynamics (`Layered`),
4. Eady dynamics (`Eady`),
5. floating Eady dynamics (`FloatingEady`),
6. two-Eady dynamics (`TwoEady`),
7. two-Eady dynamics with buoyancy jump (`TwoEadyJump`).

See `run.py` for an example of how a model is initialized and run.

This model makes use of [PyFFTW](https://pypi.python.org/pypi/pyFFTW), a Python wrapper of [FFTW](http://www.fftw.org/). Follow the [installation instructions](https://github.com/hgomersall/pyFFTW) for PyFFTW if you do not have it installed already.
