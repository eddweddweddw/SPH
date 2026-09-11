# 2-D SPH simulation of a protoplanetary disc

A smoothed-particle hydrodynamics solver for a thin accretion disc, written from
scratch in Python: kernel, density estimator, forces, artificial viscosity and
time integrator are all implemented here, with no SPH library underneath.

Built as the computational project for the *Problem Solving* course at IUSS
Pavia, alongside a master's thesis on protoplanetary disc kinematics.

![Disc evolution](figures/sph_disc_evolution.gif)

## The physical system

A razor-thin, locally isothermal disc orbiting a 1 M<sub>☉</sub> star. The gas
feels its own pressure gradient and the star's gravity; the star is treated as an
external point mass rather than as a particle. Temperature follows a power law,
T(R) = T₁₀₀ (R / 100 AU)<sup>−q</sup> with q = 0.23, so the equation of state is
P = ρ c<sub>s</sub>(R)².

Two initial conditions are available:

| `PROFILE` | Surface density | Purpose |
|---|---|---|
| `"lbp"` | Lynden-Bell & Pringle self-similar disc | Compare against the analytic viscous-spreading solution |
| `"ring"` | Thin ring at R = 200 AU | Watch viscous spreading develop from a localised feature |

The LBP profile is initialised with the pressure-corrected rotation curve, so the
disc starts close to radial equilibrium rather than relaxing into it.

## The method

| Component | Choice |
|---|---|
| Kernel | M4 cubic spline, σ<sub>2D</sub> = 10 / 7π |
| Smoothing length | Adaptive, h = η (m/ρ)<sup>1/d</sup> with η = 1.2, solved jointly with the density sum by damped fixed-point iteration |
| Neighbour search | `scipy.spatial.cKDTree`, flat pair list symmetrised so Newton's third law holds exactly |
| Pressure force | Symmetric form, P<sub>i</sub>/ρ<sub>i</sub>² + P<sub>j</sub>/ρ<sub>j</sub>² |
| Artificial viscosity | Monaghan, α = 1, β = 2, applied only on approach (**v**<sub>ij</sub>·**r**<sub>ij</sub> < 0) |
| Integrator | Leapfrog kick-drift-kick |
| Timestep | Minimum of the CFL, acceleration and viscous-signal criteria |
| Inner boundary | Accreting sink at R = 30 AU |

The inner loops are JIT-compiled with `numba`. If `numba` is not installed the
same code runs interpreted, which is considerably slower but produces identical
results.

## Validation

Two checks, both produced by the run:

**Angular momentum.** With the sink effectively disabled, specific L<sub>z</sub>
is conserved to machine precision. With an active sink the total drops, and the
run reports it against the drop expected from mass loss alone, so accretion and
numerical error can be told apart. This distinction matters: in a configuration
where the sink swallows a large fraction of the disc, the raw totals stop being
useful diagnostics and only the specific quantities mean anything.

**Energy.** Specific energy drifts slightly. The drift is expected and has two
identified sources: adaptive smoothing lengths without grad-h correction terms,
and the adaptive timestep, which breaks the symplectic property of leapfrog.

**Surface density.** For the LBP initial conditions, Σ(R) is compared against the
analytic profile, normalised to carry the same mass over the plotted range so the
comparison is like for like.

![Conservation diagnostics](figures/sph_energy_angmom.png)

## Running it

```bash
pip install -r requirements.txt
python make_ic.py      # samples the particles -> disc_ic.npz
python run_sph.py      # integrates and writes the figures
```

Parameters live in a configuration block at the top of each file. `make_ic.py`
controls the profile, particle number and disc mass; `run_sph.py` controls the
viscosity coefficients, timestep safety factors, sink radius and integration
time.

Outputs:

| File | Contents |
|---|---|
| `disc_ic_diagnostics.png` | Initial conditions: face-on view, sampled vs analytic Σ(R), rotation curve |
| `sph_density_diagnostics.png` | Σ(R) over time, per-particle density, smoothing length and neighbour count per radial bin |
| `sph_energy_angmom.png` | Energy budget and angular momentum, totals and specific values |
| `sph_disc_evolution.gif` | Animation coloured by log ρ |

## Known limitations

- No self-gravity: the disc mass enters the initial conditions but not the force
  calculation.
- No grad-h correction terms, which is the main source of the energy drift.
- 2-D only. `DIM` is carried through the kernel and density estimator, but the
  initial conditions and the diagnostics are written for the planar case.

## References

- Monaghan (1992), *Smoothed particle hydrodynamics*, ARA&A 30, 543
- Price (2012), *Smoothed particle hydrodynamics and magnetohydrodynamics*, JCP 231, 759
- Lynden-Bell & Pringle (1974), MNRAS 168, 603

## Licence

MIT.
