"""
Initial conditions for a 2-D SPH protoplanetary disc.

Samples N particles from one of two surface-density profiles, assigns an
azimuthal velocity consistent with that profile, and writes the state to
``disc_ic.npz`` for ``run_sph.py`` to pick up.

Units: distances in AU, masses in Msun, time in years, so that G = 4 pi^2.
"""

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- configuration

PROFILE = "ring"        # "lbp"  -> Lynden-Bell & Pringle self-similar disc
                        # "ring" -> thin ring centred on R_ring

# Shared parameters
Mstar = 1.0             # stellar mass                    [Msun]
Mdisc = 0.1             # total disc mass                 [Msun]
R_T   = 100.0           # temperature normalisation radius [AU]
T100  = 20.0            # temperature at R_T               [K]
q     = 0.23            # temperature power-law slope
mu    = 2.35            # mean molecular weight
N     = 1000            # number of particles
seed  = 42

# Lynden-Bell & Pringle profile
Rin   = 50.0
Rout  = 600.0
Rc    = 150.0
gamma = 1.0

# Ring profile
R_ring  = 200.0         # ring centre      [AU]
DR_ring = 20.0          # ring half-width  [AU]

# Constants (G in AU^3 Msun^-1 yr^-2; the rest in cgs for the sound speed)
G = 4.0 * np.pi**2
kB = 1.380649e-16
mp = 1.6726219e-24
AU = 1.495978707e13
YR = 3.15576e7
CMS_TO_AUYR = YR / AU


# ------------------------------------------------------------------ shared fields

def sound_speed(R):
    """Locally isothermal sound speed for a power-law T(R), in AU/yr."""
    T = T100 * (R / R_T) ** (-q)
    cs_cgs = np.sqrt(kB * T / (mu * mp))
    return cs_cgs * CMS_TO_AUYR


def v_kepler(R):
    return np.sqrt(G * Mstar / R)


# --------------------------------------------------------------------- LBP profile

def surface_density_lbp(R):
    pref = Mdisc * (2.0 - gamma) / (2.0 * np.pi * Rc**2)
    return pref * (R / Rc) ** (-gamma) * np.exp(-((R / Rc) ** (2.0 - gamma)))


def v_phi_lbp(R):
    """Keplerian rotation corrected for the radial pressure gradient."""
    cs2 = sound_speed(R) ** 2
    corr = cs2 * (1.0 + R / Rc + 2.0 * q)
    v2 = v_kepler(R) ** 2 - corr
    if np.any(v2 <= 0):
        n_bad = np.sum(v2 <= 0)
        print(f"  WARNING: {n_bad} particles with v_phi^2 <= 0; clipped to 0.")
        v2 = np.clip(v2, 0.0, None)
    return np.sqrt(v2)


def sample_radii_lbp(n, rng):
    """Inverse-transform sampling of the exponential part of the LBP profile."""
    a = np.exp(-Rin / Rc)
    b = np.exp(-Rout / Rc)
    u = rng.uniform(0.0, 1.0, n)
    return -Rc * np.log(a - u * (a - b))


def Mdisc_eff_lbp():
    return Mdisc * (np.exp(-Rin / Rc) - np.exp(-Rout / Rc))


# -------------------------------------------------------------------- ring profile

def surface_density_ring(R):
    Sigma0 = Mdisc / (np.pi * ((R_ring + DR_ring) ** 2 - (R_ring - DR_ring) ** 2))
    return np.where(np.abs(R - R_ring) < DR_ring, Sigma0, 0.0)


def v_phi_ring(R):
    return v_kepler(R)


def sample_radii_ring(n, rng):
    u = rng.uniform(0.0, 1.0, n)
    Rin2 = (R_ring - DR_ring) ** 2
    Rout2 = (R_ring + DR_ring) ** 2
    return np.sqrt(u * (Rout2 - Rin2) + Rin2)


def Mdisc_eff_ring():
    return Mdisc


if PROFILE == "lbp":
    sample_radii = sample_radii_lbp
    surface_density = surface_density_lbp
    v_phi = v_phi_lbp
    Mdisc_eff = Mdisc_eff_lbp
    R_inner_plot, R_outer_plot = Rin, Rout
elif PROFILE == "ring":
    sample_radii = sample_radii_ring
    surface_density = surface_density_ring
    v_phi = v_phi_ring
    Mdisc_eff = Mdisc_eff_ring
    R_inner_plot, R_outer_plot = R_ring - 2 * DR_ring, R_ring + 2 * DR_ring
else:
    raise ValueError(f"PROFILE='{PROFILE}' not recognised. Use 'lbp' or 'ring'.")


# ------------------------------------------------------------ build the particles

def build_ic():
    rng = np.random.default_rng(seed)

    R = sample_radii(N, rng)
    phi = rng.uniform(0.0, 2.0 * np.pi, N)
    x = R * np.cos(phi)
    y = R * np.sin(phi)

    Meff = Mdisc_eff()
    m = np.full(N, Meff / N)

    vphi = v_phi(R)
    vx = -vphi * np.sin(phi)
    vy = vphi * np.cos(phi)

    print(f"Profile: {PROFILE.upper()}")
    print(f"  N = {N}, effective mass = {Meff:.4f} Msun "
          f"(mass per particle = {m[0]:.3e})")

    # Seed the smoothing length from the median nearest-neighbour distance.
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    h_initial = 1.2 * np.median(d[:, 1])
    print(f"  Median nearest-neighbour distance = {np.median(d[:, 1]):.2f} AU")
    print(f"  -> initial h = {h_initial:.2f} AU")

    return dict(x=x, y=y, vx=vx, vy=vy, m=m, R=R, phi=phi, vphi=vphi, h=h_initial)


# ------------------------------------------------------------------- diagnostics

def plot_diagnostics(ic, fname="disc_ic_diagnostics.png"):
    """Face-on view, sampled vs analytic surface density, rotation curve."""
    x, y, R, vphi, m = ic["x"], ic["y"], ic["R"], ic["vphi"], ic["m"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

    ax[0].scatter(x, y, s=3, c=R, cmap="viridis", alpha=0.7)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [AU]")
    ax[0].set_ylabel("y [AU]")
    ax[0].set_title(f"Initial conditions, face-on ({PROFILE})")

    if PROFILE == "lbp":
        bins = np.logspace(np.log10(R_inner_plot), np.log10(R_outer_plot), 30)
    else:
        bins = np.linspace(R_inner_plot, R_outer_plot, 30)
    counts, edges = np.histogram(R, bins=bins)
    area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    Sigma_meas = counts * m[0] / area
    Rmid = 0.5 * (edges[1:] + edges[:-1])
    Rfine = np.linspace(R_inner_plot, R_outer_plot, 200)

    if PROFILE == "lbp":
        ax[1].loglog(Rmid, Sigma_meas, "o", ms=4, label="sampled")
        ax[1].loglog(Rfine, surface_density(Rfine), "-", label="analytic")
    else:
        ax[1].plot(Rmid, Sigma_meas, "o", ms=4, label="sampled")
        ax[1].plot(Rfine, surface_density(Rfine), "-", label="analytic (box)")
    ax[1].set_xlabel("R [AU]")
    ax[1].set_ylabel(r"$\Sigma$ [Msun/AU$^2$]")
    ax[1].set_title("Surface density")
    ax[1].legend()

    order = np.argsort(R)
    ax[2].plot(R[order], vphi[order], ".", ms=4, alpha=0.4, color="red",
               label=r"$v_\phi$, particles")
    ax[2].plot(Rfine, v_kepler(Rfine), "--", label=r"$v_K$ (Keplerian)")
    if PROFILE == "lbp":
        ax[2].plot(Rfine, v_phi(Rfine), "-", lw=2,
                   label=r"$v_\phi$ with pressure support")
    ax[2].set_xlabel("R [AU]")
    ax[2].set_ylabel("v [AU/yr]")
    ax[2].set_title("Rotation curve")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    print(f"  Figure written to {fname}")


if __name__ == "__main__":
    ic = build_ic()
    plot_diagnostics(ic)
    np.savez("disc_ic.npz",
             x=ic["x"], y=ic["y"], vx=ic["vx"], vy=ic["vy"], m=ic["m"],
             h=ic["h"], Mstar=Mstar, G=G,
             Rc=Rc, R0=R_T, T100=T100, q=q, mu=mu)
    print("  Initial state written to disc_ic.npz")
