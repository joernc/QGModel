# QGModel
This is Python code that solves the nonlinear QG equations for PV anomalies on a doubly-periodic domain.  It consists of two parts:

1. a fully-dealiased pseudo-spectral code that steps forward the PV conservation equations and
2. implementations of specific model types that provide an inversion relation.

At the moment, five model types are supported:

1. two-dimensional dynamics (`TwoDim`),
2. two-layer dynamics (`TwoLayer`),
3. Eady dynamics (`Eady`),
4. floating Eady dynamics (`FloatingEady`),
5. two-Eady dynamics (`TwoEady`).

See `run.py` for an example of how a model is initialized and run.
