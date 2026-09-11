"""
2-D SPH integration of a protoplanetary disc.

Reads ``disc_ic.npz`` (written by ``make_ic.py``), integrates the disc with a
leapfrog KDK scheme, and writes density, conservation and animation diagnostics.

Method
------
* M4 cubic-spline kernel, sigma_2D = 10 / (7 pi)
* Adaptive smoothing length from h = eta (m / rho)^(1/d), solved by damped
  fixed-point iteration jointly with the density sum
* Symmetric pressure force, P_i/rho_i^2 + P_j/rho_j^2
* Monaghan artificial viscosity, alpha = 1, beta = 2, applied on approach only
* Leapfrog kick-drift-kick with a timestep set by the minimum of CFL,
  acceleration and viscous-signal criteria
* Locally isothermal equation of state, P = rho cs(R)^2
* Accreting sink at R_SINK; the central star is an external point mass

Units: AU, Msun, yr (so G = 4 pi^2).
"""

import time

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# numba is optional: without it the same code runs interpreted, much more slowly.
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    prange = range

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


# ---------------------------------------------------------------- configuration

DIM = 2

ALPHA_AV = 1.0          # Monaghan artificial viscosity, linear term
BETA_AV = 2.0           # Monaghan artificial viscosity, quadratic term
EPS_AV = 0.01           # softening in mu_ij, in units of h^2

ETA = 1.2               # smoothing length: h = ETA (m / rho)^(1/DIM)
H_MIN, H_MAX = 0.5, 200.0

C_CFL, C_ACC, C_VISC = 0.3, 0.25, 0.25   # timestep safety factors

R_SINK = 30.0           # accreting inner boundary [AU]

T_END = 5000.0          # integration time  [yr]
SNAP_EVERY = 50.0       # snapshot interval [yr]
PRINT_EVERY_STEP = 100

# Physical constants (cgs, for the sound speed)
kB = 1.380649e-16
mp_cgs = 1.6726219e-24
AU = 1.495978707e13
YR = 3.15576e7
CMS_TO_AUYR = YR / AU


# ------------------------------------------------------------------------ kernel

@njit(inline="always", fastmath=True)
def W_value(r, h, dim):
    """M4 cubic spline."""
    if h <= 0.0:
        return 0.0
    q = r / h
    if q < 1.0:
        f = 1.0 - 1.5 * q * q + 0.75 * q * q * q
    elif q < 2.0:
        f = 0.25 * (2.0 - q) ** 3
    else:
        return 0.0
    sigma = 10.0 / (7.0 * np.pi) if dim == 2 else 1.0 / np.pi
    return sigma * f / (h ** dim)


@njit(inline="always", fastmath=True)
def dW_value(r, h, dim):
    """Radial derivative of the M4 cubic spline."""
    if h <= 0.0 or r <= 0.0:
        return 0.0
    q = r / h
    if q < 1.0:
        f = -3.0 * q + 2.25 * q * q
    elif q < 2.0:
        f = -0.75 * (2.0 - q) ** 2
    else:
        return 0.0
    sigma = 10.0 / (7.0 * np.pi) if dim == 2 else 1.0 / np.pi
    return sigma * f / (h ** (dim + 1))


# ------------------------------------------------------------------- neighbours

def build_pairs(pos, h):
    """Flat, symmetrised pair list within 2 max(h_i, h_j), so Newton's third law holds."""
    tree = cKDTree(pos)
    neigh = tree.query_ball_point(pos, r=2.0 * h)
    counts = np.fromiter((len(n) for n in neigh), dtype=np.int64, count=len(neigh))
    M = counts.sum()
    pair_i = np.empty(M, dtype=np.int64)
    pair_j = np.empty(M, dtype=np.int64)
    k = 0
    for i, lst in enumerate(neigh):
        L = len(lst)
        if L:
            pair_i[k:k + L] = i
            pair_j[k:k + L] = lst
            k += L
    Nmax = pos.shape[0]
    code1 = pair_i * Nmax + pair_j
    code2 = pair_j * Nmax + pair_i
    code_union = np.unique(np.concatenate([code1, code2]))
    return code_union // Nmax, code_union % Nmax


# ---------------------------------------------------------------------- density

@njit(parallel=True, fastmath=True)
def density_sum(pos, m, h, pair_i, pair_j, dim):
    N = pos.shape[0]
    rho = np.zeros(N)
    for k in prange(pair_i.size):
        i = pair_i[k]
        j = pair_j[k]
        s = 0.0
        for d in range(dim):
            dx = pos[i, d] - pos[j, d]
            s += dx * dx
        r = np.sqrt(s)
        if r < 2.0 * h[i]:
            rho[i] += m[j] * W_value(r, h[i], dim)
    return rho


@njit(parallel=True, fastmath=True)
def count_neighbours(pos, h, pair_i, pair_j, dim):
    """Number of neighbours within 2 h_i for each particle."""
    N = pos.shape[0]
    nn = np.zeros(N, dtype=np.int64)
    for k in prange(pair_i.size):
        i = pair_i[k]
        j = pair_j[k]
        s = 0.0
        for d in range(dim):
            dx = pos[i, d] - pos[j, d]
            s += dx * dx
        r = np.sqrt(s)
        if r < 2.0 * h[i]:
            nn[i] += 1
    return nn


def compute_density(pos, m, h, n_iter=4):
    """Damped fixed-point iteration for (rho, h) with h = ETA (m/rho)^(1/DIM)."""
    rho = np.zeros(pos.shape[0])
    for _ in range(n_iter):
        pair_i, pair_j = build_pairs(pos, h)
        rho = density_sum(pos, m, h, pair_i, pair_j, DIM)
        h_new = ETA * np.power(m / np.maximum(rho, 1e-30), 1.0 / DIM)
        h_new = np.clip(h_new, H_MIN, H_MAX)
        h = 0.5 * (h + h_new)
    return rho, h, pair_i, pair_j


# ----------------------------------------------------------------- acceleration

@njit(parallel=True, fastmath=True)
def accel_pressure_viscosity(pos, vel, m, rho, h, cs, P,
                             pair_i, pair_j, dim,
                             alpha_av, beta_av, eps_av):
    N = pos.shape[0]
    acc = np.zeros((N, dim))
    visc_max_per_pair = np.zeros(pair_i.size)

    for k in prange(pair_i.size):
        i = pair_i[k]
        j = pair_j[k]
        if i == j:
            continue
        rij = np.empty(dim)
        s = 0.0
        for d in range(dim):
            rij[d] = pos[i, d] - pos[j, d]
            s += rij[d] * rij[d]
        r = np.sqrt(s)
        if r <= 0.0:
            continue
        hij = 0.5 * (h[i] + h[j])
        if r >= 2.0 * hij:
            continue
        dWdr = dW_value(r, hij, dim)
        vdotr = 0.0
        for d in range(dim):
            vdotr += (vel[i, d] - vel[j, d]) * rij[d]
        Pi = 0.0
        if vdotr < 0.0:                     # viscosity acts on approach only
            mu_ij = hij * vdotr / (r * r + eps_av * hij * hij)
            cs_avg = 0.5 * (cs[i] + cs[j])
            rho_avg = 0.5 * (rho[i] + rho[j])
            Pi = (-alpha_av * cs_avg * mu_ij + beta_av * mu_ij * mu_ij) / rho_avg
            visc_max_per_pair[k] = abs(mu_ij)
        coef = P[i] / (rho[i] * rho[i]) + P[j] / (rho[j] * rho[j]) + Pi
        fac = -m[j] * coef * dWdr / r
        for d in range(dim):
            acc[i, d] += fac * rij[d]
    visc_max = visc_max_per_pair.max() if visc_max_per_pair.size > 0 else 0.0
    return acc, visc_max


def compute_acceleration(pos, vel, m, rho, h, cs, cs2, pair_i, pair_j):
    """SPH pressure and viscosity, plus the central star as an external point mass."""
    P = rho * cs2
    acc, visc_max = accel_pressure_viscosity(
        pos, vel, m, rho, h, cs, P, pair_i, pair_j, DIM,
        ALPHA_AV, BETA_AV, EPS_AV,
    )
    R2 = np.sum(pos * pos, axis=1)
    R3 = R2 * np.sqrt(R2)
    for d in range(DIM):
        acc[:, d] -= G * Mstar * pos[:, d] / R3
    return acc, visc_max


# ---------------------------------------------------- timestep, sink, diagnostics

def sound_speed(R):
    T = T100 * (R / R0) ** (-q)
    cs_cgs = np.sqrt(kB * T / (mu * mp_cgs))
    return cs_cgs * CMS_TO_AUYR


def choose_dt(h, cs, acc, visc_max):
    """Minimum of the CFL, acceleration and viscous-signal timesteps."""
    dt_cfl = C_CFL * np.min(h / cs)
    a = np.sqrt(np.sum(acc * acc, axis=1))
    dt_acc = C_ACC * np.min(np.sqrt(h / np.maximum(a, 1e-30)))
    signal = (1.0 + 1.5 * ALPHA_AV) * cs.max() + 1.5 * BETA_AV * visc_max
    dt_visc = C_VISC * h.min() / max(signal, 1e-30)
    return min(dt_cfl, dt_acc, dt_visc)


def apply_sink(pos, vel, m, h):
    """Remove particles that cross the inner boundary."""
    R2 = np.sum(pos * pos, axis=1)
    keep = R2 > R_SINK ** 2
    return pos[keep], vel[keep], m[keep], h[keep], np.sum(~keep)


def disc_diagnostics(pos, vel, m):
    R = np.sqrt(np.sum(pos * pos, axis=1))
    cs2 = sound_speed(R) ** 2
    Ekin = 0.5 * np.sum(m * np.sum(vel * vel, axis=1))
    Egrav = -np.sum(G * Mstar * m / R)
    Etherm = np.sum(m * cs2)
    Lz = np.sum(m * (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]))
    return dict(Ekin=Ekin, Egrav=Egrav, Etherm=Etherm,
                Etot=Ekin + Egrav + Etherm, Lz=Lz, Mtot=np.sum(m))


# ------------------------------------------------------------------------- main

def load_ic(path="disc_ic.npz"):
    global Mstar, G, Rc, R0, T100, q, mu
    data = np.load(path)
    N = len(data["x"])

    pos = np.zeros((N, DIM))
    vel = np.zeros((N, DIM))
    pos[:, 0] = data["x"]
    pos[:, 1] = data["y"]
    vel[:, 0] = data["vx"]
    vel[:, 1] = data["vy"]

    m = data["m"].copy()
    h = np.full(N, float(data["h"]))

    Mstar = float(data["Mstar"])
    G = float(data["G"])
    Rc = float(data["Rc"])
    R0 = float(data["R0"])
    T100 = float(data["T100"])
    q = float(data["q"])
    mu = float(data["mu"])

    print(f"Initial conditions loaded (DIM={DIM}): N = {N}, "
          f"total mass = {m.sum():.4f} Msun")
    if not HAVE_NUMBA:
        print("  numba not found: running interpreted, expect this to be slow.")
    return pos, vel, m, h


def integrate(pos, vel, m, h):
    print(f"Artificial viscosity: alpha={ALPHA_AV}, beta={BETA_AV}")
    print("Step  t [yr]   dt [yr]   N    <rho>          <h> [AU]")

    R = np.sqrt(np.sum(pos * pos, axis=1))
    cs = sound_speed(R)
    cs2 = cs * cs
    rho, h, pair_i, pair_j = compute_density(pos, m, h)
    acc, visc_max = compute_acceleration(pos, vel, m, rho, h, cs, cs2, pair_i, pair_j)

    snapshots, diag_t, diag_data = [], [], []
    nn0 = count_neighbours(pos, h, pair_i, pair_j, DIM)
    snapshots.append(dict(t=0.0, pos=pos.copy(), vel=vel.copy(),
                          rho=rho.copy(), h=h.copy(), n_neigh=nn0.copy()))
    diag_t.append(0.0)
    diag_data.append(disc_diagnostics(pos, vel, m))

    t, step, t_next_snap = 0.0, 0, SNAP_EVERY
    t0 = time.time()
    while t < T_END:
        dt = choose_dt(h, cs, acc, visc_max)
        if t + dt > T_END:
            dt = T_END - t

        vel += 0.5 * dt * acc                                  # first kick
        pos += dt * vel                                        # drift
        pos, vel, m, h, _ = apply_sink(pos, vel, m, h)

        R = np.sqrt(np.sum(pos * pos, axis=1))
        cs = sound_speed(R)
        cs2 = cs * cs
        rho, h, pair_i, pair_j = compute_density(pos, m, h)
        acc, visc_max = compute_acceleration(pos, vel, m, rho, h, cs, cs2,
                                             pair_i, pair_j)
        vel += 0.5 * dt * acc                                  # second kick

        t += dt
        step += 1
        if step % PRINT_EVERY_STEP == 0 or step == 1:
            print(f"{step:4d}  {t:7.2f}  {dt:7.4f}  {len(m):4d}   "
                  f"{rho.mean():.3e}    {h.mean():6.2f}")

        if t >= t_next_snap - 1e-9:
            nn = count_neighbours(pos, h, pair_i, pair_j, DIM)
            snapshots.append(dict(t=t, pos=pos.copy(), vel=vel.copy(),
                                  rho=rho.copy(), h=h.copy(), n_neigh=nn))
            diag_t.append(t)
            diag_data.append(disc_diagnostics(pos, vel, m))
            t_next_snap += SNAP_EVERY

    elapsed = time.time() - t0
    print(f"\nDone. {step} steps in {elapsed:.1f} s "
          f"({elapsed / step * 1000:.1f} ms/step).")
    print(f"Snapshots kept: {len(snapshots)}.")
    return snapshots, np.array(diag_t), diag_data, m


# --------------------------------------------------------- density diagnostics

def plot_density_diagnostics(snapshots, m, fname="sph_density_diagnostics.png"):
    print("\nWriting density diagnostics...")

    n_show = 4
    idx_show = np.linspace(0, len(snapshots) - 1, n_show, dtype=int)
    selected = [snapshots[i] for i in idx_show]
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_show))

    R_final = np.sqrt(np.sum(selected[-1]["pos"] * selected[-1]["pos"], axis=1))
    R_min = max(R_SINK, R_final.min())
    R_max = R_final.max() * 1.02
    bins = np.logspace(np.log10(R_min), np.log10(R_max), 30)
    Rmid = 0.5 * (bins[1:] + bins[:-1])

    # LBP reference, normalised to the mass the initial conditions put in
    # [R_min, R_max] so the comparison is like for like.
    GAMMA_LBP = 1.0
    m0 = m.mean()
    # numpy 2.0 removed np.trapz, so resolve lazily rather than with getattr's
    # eagerly-evaluated default, which would raise on numpy >= 2.0.
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    def lbp_shape(R, Rc_local, gamma=GAMMA_LBP):
        return (R / Rc_local) ** (-gamma) * np.exp(-((R / Rc_local) ** (2.0 - gamma)))

    R_init = np.sqrt(np.sum(snapshots[0]["pos"] * snapshots[0]["pos"], axis=1))
    M_ref = m0 * np.count_nonzero((R_init >= R_min) & (R_init <= R_max))
    Rg = np.logspace(np.log10(R_min), np.log10(R_max), 4000)
    C_lbp = M_ref / trapz(lbp_shape(Rg, Rc) * 2.0 * np.pi * Rg, Rg)

    def Sigma_LBP_ref(R):
        return C_lbp * lbp_shape(R, Rc)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for s, c in zip(selected, colors):
        Rs = np.sqrt(np.sum(s["pos"] * s["pos"], axis=1))
        counts, _ = np.histogram(Rs, bins=bins)
        area = np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
        ax1.loglog(Rmid, counts * m0 / area, "o-", ms=4, color=c, lw=1.2,
                   label=f"t = {s['t']:.0f} yr")
    Rfine = np.logspace(np.log10(R_min), np.log10(R_max), 200)
    ax1.loglog(Rfine, Sigma_LBP_ref(Rfine), "k--", lw=2, alpha=0.7,
               label="LBP analytic\n(same mass in range, Rc)")
    ax1.set_xlabel("R [AU]")
    ax1.set_ylabel(r"$\Sigma$ [Msun/AU$^2$]")
    ax1.set_title("Surface density over time")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    ax2 = axes[0, 1]
    s_last = selected[-1]
    R_last = np.sqrt(np.sum(s_last["pos"] * s_last["pos"], axis=1))
    ax2.scatter(R_last, s_last["rho"], s=4, alpha=0.5, c=R_last, cmap="viridis")
    ax2.set_xlabel("R [AU]")
    ax2.set_ylabel(r"$\rho_{\rm SPH}$ [Msun/AU$^2$]")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_title(f"Per-particle SPH density at t = {s_last['t']:.0f} yr")
    ax2.grid(alpha=0.3, which="both")

    ax3 = axes[1, 0]
    for s, c in zip(selected, colors):
        Rs = np.sqrt(np.sum(s["pos"] * s["pos"], axis=1))
        h_mean = np.full(len(Rmid), np.nan)
        for k in range(len(Rmid)):
            sel = (Rs >= bins[k]) & (Rs < bins[k + 1])
            if sel.any():
                h_mean[k] = np.mean(s["h"][sel])
        ax3.loglog(Rmid, h_mean, "o-", ms=4, color=c, lw=1.2,
                   label=f"t = {s['t']:.0f} yr")
    ax3.set_xlabel("R [AU]")
    ax3.set_ylabel(r"$\langle h \rangle$ [AU]")
    ax3.set_title("Mean smoothing length per radial bin")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3, which="both")

    ax4 = axes[1, 1]
    for s, c in zip(selected, colors):
        Rs = np.sqrt(np.sum(s["pos"] * s["pos"], axis=1))
        nmean = np.full(len(Rmid), np.nan)
        for k in range(len(Rmid)):
            sel = (Rs >= bins[k]) & (Rs < bins[k + 1])
            if sel.any():
                nmean[k] = np.mean(s["n_neigh"][sel])
        ax4.semilogx(Rmid, nmean, "o-", ms=4, color=c, lw=1.2,
                     label=f"t = {s['t']:.0f} yr")
    ax4.set_xlabel("R [AU]")
    ax4.set_ylabel(r"$\langle N_{\rm neigh}\rangle$")
    ax4.set_title("Neighbour count per radial bin")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written to {fname}")


# ------------------------------------------------- conservation diagnostics

def plot_conservation(diag_t, diag_data, fname="sph_energy_angmom.png"):
    Etot = np.array([d["Etot"] for d in diag_data])
    Lz = np.array([d["Lz"] for d in diag_data])
    Mtot = np.array([d["Mtot"] for d in diag_data])
    Ekin = np.array([d["Ekin"] for d in diag_data])
    Egrav = np.array([d["Egrav"] for d in diag_data])
    Etherm = np.array([d["Etherm"] for d in diag_data])
    Etot_spec = Etot / Mtot
    Lz_spec = Lz / Mtot

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(diag_t, Ekin, label="kinetic")
    axes[0, 0].plot(diag_t, Egrav, label="gravitational")
    axes[0, 0].plot(diag_t, Etherm, label="thermal")
    axes[0, 0].plot(diag_t, Etot, "k-", lw=2, label="total")
    axes[0, 0].set_xlabel("t [yr]")
    axes[0, 0].set_ylabel("E")
    axes[0, 0].set_title("Energy budget")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(diag_t, Etot_spec / Etot_spec[0] - 1, "k-")
    axes[0, 1].axhline(0, color="gray", ls=":", lw=0.7)
    axes[0, 1].set_xlabel("t [yr]")
    axes[0, 1].set_ylabel(r"$\Delta E_{\rm spec}/E_{\rm spec}(0)$")
    axes[0, 1].set_title("Specific energy drift")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(diag_t, Lz, "b-", label=r"total $L_z$")
    axes[1, 0].plot(diag_t, Mtot * Lz_spec[0], "b--", alpha=0.5,
                    label="expected from mass loss alone")
    axes[1, 0].set_xlabel("t [yr]")
    axes[1, 0].set_ylabel(r"$L_z$")
    axes[1, 0].set_title("Total angular momentum")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(diag_t, Lz_spec / Lz_spec[0] - 1, "b-")
    axes[1, 1].axhline(0, color="gray", ls=":", lw=0.7)
    axes[1, 1].set_xlabel("t [yr]")
    axes[1, 1].set_ylabel(r"$\Delta L_{z,\rm spec}/L_{z,\rm spec}(0)$")
    axes[1, 1].set_title("Specific angular momentum drift")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written to {fname}")

    print("\nFinal relative drift:")
    print(f"  specific energy:            {(Etot_spec[-1] / Etot_spec[0] - 1):+.3e}")
    print(f"  specific angular momentum:  {(Lz_spec[-1] / Lz_spec[0] - 1):+.3e}")
    print(f"  mass accreted by the sink:  {(1 - Mtot[-1] / Mtot[0]) * 100:.2f}%")


# -------------------------------------------------------------------- animation

def make_animation(snapshots, fname="sph_disc_evolution.gif", max_frames=120):
    print("\nWriting animation...")
    all_R = np.concatenate(
        [np.sqrt(np.sum(s["pos"] * s["pos"], axis=1)) for s in snapshots])
    LIM = 1.1 * np.max(all_R)

    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")

    def color_from_rho(rho_):
        return np.log10(np.maximum(rho_, 1e-12))

    all_logrho = np.concatenate([color_from_rho(s["rho"]) for s in snapshots])
    vmin, vmax = np.percentile(all_logrho, [2, 98])

    scat = ax.scatter(snapshots[0]["pos"][:, 0], snapshots[0]["pos"][:, 1],
                      c=color_from_rho(snapshots[0]["rho"]),
                      s=4, cmap="viridis", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(scat, ax=ax, fraction=0.046)
    cbar.set_label(r"$\log_{10}\rho$")
    title = ax.set_title(
        f"t = {snapshots[0]['t']:.0f} yr,  N = {len(snapshots[0]['pos'])}")

    def update(frame):
        s = snapshots[frame]
        scat.set_offsets(s["pos"][:, :2])
        scat.set_array(color_from_rho(s["rho"]))
        title.set_text(f"t = {s['t']:.0f} yr,  N = {len(s['pos'])}")
        return scat, title

    stride = max(1, len(snapshots) // max_frames)
    frames = list(range(0, len(snapshots), stride))
    if frames[-1] != len(snapshots) - 1:
        frames.append(len(snapshots) - 1)
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=80, blit=False)
    anim.save(fname, writer=animation.PillowWriter(fps=12))
    plt.close(fig)
    print(f"  Written to {fname} ({len(frames)} frames).")


if __name__ == "__main__":
    pos, vel, m, h = load_ic()
    snapshots, diag_t, diag_data, m = integrate(pos, vel, m, h)
    plot_density_diagnostics(snapshots, m)
    plot_conservation(diag_t, diag_data)
    make_animation(snapshots)
