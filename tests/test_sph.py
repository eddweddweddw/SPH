"""Checks of the properties the solver relies on: kernel normalisation and
pairwise antisymmetry of the SPH forces (which is what conserves momentum and
angular momentum exactly)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_sph  # noqa: E402


def test_kernel_normalised_in_2d():
    h = 3.0
    r = np.linspace(0.0, 2.0 * h, 20001)
    W = np.array([run_sph.W_value(x, h, 2) for x in r])
    integral = np.trapezoid(W * 2.0 * np.pi * r, r)
    assert abs(integral - 1.0) < 1e-6


def test_kernel_derivative_matches_finite_difference():
    h = 2.0
    for r in (0.3, 1.1, 2.5, 3.7):
        eps = 1e-6
        fd = (run_sph.W_value(r + eps, h, 2) - run_sph.W_value(r - eps, h, 2)) / (2 * eps)
        assert abs(run_sph.dW_value(r, h, 2) - fd) < 1e-6


def _random_gas(n=300, seed=1):
    rng = np.random.default_rng(seed)
    R = rng.uniform(50.0, 150.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.column_stack([R * np.cos(phi), R * np.sin(phi)])
    vel = rng.normal(0.0, 0.5, (n, 2))      # random, so viscosity switches on
    m = np.full(n, 1e-4)
    h = np.full(n, 8.0)
    cs = rng.uniform(0.05, 0.1, n)
    return pos, vel, m, h, cs


def _sph_forces():
    pos, vel, m, h, cs = _random_gas()
    pair_i, pair_j = run_sph.build_pairs(pos, h)
    rho = run_sph.density_sum(pos, m, h, pair_i, pair_j, 2)
    acc, _ = run_sph.accel_pressure_viscosity(
        pos, vel, m, rho, h, cs, rho * cs ** 2, pair_i, pair_j, 2,
        run_sph.ALPHA_AV, run_sph.BETA_AV, run_sph.EPS_AV)
    return pos, m, acc


def test_sph_forces_conserve_linear_momentum():
    pos, m, acc = _sph_forces()
    F = m[:, None] * acc
    assert np.all(np.abs(F.sum(axis=0)) < 1e-12 * np.abs(F).sum())


def test_sph_forces_exert_no_net_torque():
    pos, m, acc = _sph_forces()
    torque = m * (pos[:, 0] * acc[:, 1] - pos[:, 1] * acc[:, 0])
    assert abs(torque.sum()) < 1e-12 * np.abs(torque).sum()
