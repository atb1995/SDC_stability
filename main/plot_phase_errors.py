"""
plot_phase_error.py

Plot phase error R = theta / (omega * dt) along the imaginary axis
for implicit and explicit components of IMEX SDC and RK schemes.

R = 1  => exact phase speed
R > 1  => phase-leading (too fast)
R < 1  => phase-lagging (too slow)

Produces one figure per scheme group per component:
  Implicit: RK, SDC-LU, SDC-MIN-SR-S, SDC-MIN-SR-FLEX
  Explicit: RK, SDC-PIC, SDC-FE
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmat import genQCoeffs, genQDeltaCoeffs
from stability_functions import build_imex_rk_schemes, sdc_stability_imex_grid

DEFAULT_MK_LIST = [(2, [1, 2, 3]), (3, [2, 3, 4]), (4, [3, 4, 5]), (5, [4, 5, 6])]

COLORS = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def phase_ratio(A_vals, odt):
    """R = unwrapped_theta / (omega*dt). Smooth, no branch-cut jumps."""
    theta = np.unwrap(np.arctan2(np.imag(A_vals), np.real(A_vals)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(odt) > 1e-14, theta / odt, np.nan)


def _imex_rk_stability_complex_grid(z_imp_grid, z_exp_grid, a_imp, a_exp, b_imp, b_exp):
    nu = len(b_imp)
    shape = z_imp_grid.shape
    stages = np.zeros(shape + (nu,), dtype=complex)
    for j in range(nu):
        rhs = np.ones(shape, dtype=complex)
        for l in range(j):
            rhs += z_exp_grid * a_exp[j, l] * stages[..., l]
            rhs += z_imp_grid * a_imp[j, l] * stages[..., l]
        stages[..., j] = rhs / (1.0 - z_imp_grid * a_imp[j, j])
    result = np.ones(shape, dtype=complex)
    for j in range(nu):
        result += z_exp_grid * b_exp[j] * stages[..., j]
        result += z_imp_grid * b_imp[j] * stages[..., j]
    return result


def sdc_amplification(odt_vals, nodes, weights, qmat, qdelta_imp, qdelta_exp,
                      k, component, quad_update,
                      qdelta_imp_type=None, qdelta_imp_seq=None):
    """Amplification factor along z = i*odt for one SDC config."""
    z = 1j * odt_vals
    zero = np.zeros_like(z)
    dtf, dts = (z, zero) if component == "implicit" else (zero, z)
    return sdc_stability_imex_grid(
        dtf[np.newaxis, :], dts[np.newaxis, :],
        nodes, weights, qmat, qdelta_imp, qdelta_exp, k,
        quad_update=quad_update,
        qdelta_imp_type=qdelta_imp_type,
        qdelta_imp_seq=qdelta_imp_seq,
    )[0]


def rk_amplification(odt_vals, scheme, component):
    """Amplification factor along z = i*odt for one RK scheme."""
    z = 1j * odt_vals
    zero = np.zeros_like(z)
    z_imp, z_exp = (z, zero) if component == "implicit" else (zero, z)
    return _imex_rk_stability_complex_grid(
        z_imp[np.newaxis, :], z_exp[np.newaxis, :],
        scheme["A_imp"], scheme["A_exp"],
        scheme["b_imp"], scheme["b_exp"],
    )[0]


# ---------------------------------------------------------------------------
# Curve builders — return list of (label, R_array)
# ---------------------------------------------------------------------------

def rk_curves(odt_vals, component):
    curves = []
    for name, scheme in build_imex_rk_schemes().items():
        A = rk_amplification(odt_vals, scheme, component)
        curves.append((name, phase_ratio(A, odt_vals)))
    return curves


def sdc_curves(odt_vals, component, qdelta_imp_type, qdelta_exp_type,
               node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=True):
    curves = []
    for m, ks in DEFAULT_MK_LIST:
        nodes, weights, qmat = genQCoeffs(
            "Collocation", nNodes=m, nodeType=node_type, quadType=quad_type
        )
        qdelta_exp = genQDeltaCoeffs(qdelta_exp_type, nodes=nodes, Q=qmat)
        for k in ks:
            qdelta_imp_seq = None
            qdelta_imp = None
            if qdelta_imp_type == "MIN-SR-FLEX":
                qdelta_imp_seq = [
                    genQDeltaCoeffs(
                        "MIN-SR-FLEX", nNodes=m, nodeType=node_type,
                        quadType=quad_type, nodes=nodes, Q=qmat, k=sweep,
                    )
                    for sweep in range(1, k + 1)
                ]
            else:
                qdelta_imp = genQDeltaCoeffs(
                    qdelta_imp_type, nNodes=m, nodeType=node_type,
                    quadType=quad_type, nodes=nodes, Q=qmat,
                )
            A = sdc_amplification(
                odt_vals, nodes, weights, qmat,
                qdelta_imp, qdelta_exp, k, component, quad_update,
                qdelta_imp_type=qdelta_imp_type if qdelta_imp_seq is not None else None,
                qdelta_imp_seq=qdelta_imp_seq,
            )
            curves.append((f"M={m} K={k}", phase_ratio(A, odt_vals)))
    return curves


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_phase_group(odt_vals, curves, title, odt_max, ylim=(-0.2, 1.4)):
    """One figure, one panel per M value (4 rows), all K overlaid per group."""
    # Group curves by M (SDC) or just plot all together (RK)
    # curves is a flat list of (label, R); for RK all go on every panel,
    # for SDC we split by M.
    is_sdc = any("M=" in label for label, _ in curves)

    if is_sdc:
        # Split into groups by M
        m_groups = {}
        for label, R in curves:
            m = label.split(" ")[0]  # "M=2"
            m_groups.setdefault(m, []).append((label, R))
        n_rows = len(m_groups)
        fig, axes = plt.subplots(n_rows, 1, figsize=(9, 3.5 * n_rows), squeeze=False)
        for row, (m_label, mcurves) in enumerate(m_groups.items()):
            ax = axes[row][0]
            _draw_panel(ax, odt_vals, mcurves, f"{title} — {m_label}", odt_max, ylim)
    else:
        # RK: single panel
        fig, axes = plt.subplots(1, 1, figsize=(9, 4.5), squeeze=False)
        _draw_panel(axes[0][0], odt_vals, curves, title, odt_max, ylim)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def _draw_panel(ax, odt_vals, curves, title, odt_max, ylim):
    ax.axhline(1.0, color="black", linewidth=1.4, linestyle="--", label="Exact (R=1)", zorder=3)
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    for i, (label, R) in enumerate(curves):
        # Mask R outside ylim so runaway unstable values don't compress the plot
        R_plot = np.where((R > ylim[0] - 0.5) & (R < ylim[1] + 0.5), R, np.nan)
        ax.plot(odt_vals, R_plot, linewidth=1.5, label=label, color=COLORS[i % len(COLORS)])
    ax.set_xlim(0, odt_max)
    ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\omega \Delta t$", fontsize=10)
    ax.set_ylabel(r"Phase ratio $R$", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(True, linewidth=0.4, alpha=0.4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

IMPLICIT_GROUPS = [
    ("RK",          dict(scheme_type="rk")),
    ("SDC-LU",      dict(scheme_type="sdc", qdelta_imp_type="LU",         qdelta_exp_type="FE")),
    ("SDC-MIN-SR-S",    dict(scheme_type="sdc", qdelta_imp_type="MIN-SR-S",    qdelta_exp_type="FE")),
    ("SDC-MIN-SR-FLEX", dict(scheme_type="sdc", qdelta_imp_type="MIN-SR-FLEX", qdelta_exp_type="FE")),
]

EXPLICIT_GROUPS = [
    ("RK",      dict(scheme_type="rk")),
    ("SDC-PIC", dict(scheme_type="sdc", qdelta_imp_type="LU", qdelta_exp_type="PIC")),
    ("SDC-FE",  dict(scheme_type="sdc", qdelta_imp_type="LU", qdelta_exp_type="FE")),
]


def run(component, n=400, odt_max=None, quad_update=True,
        node_type="LEGENDRE", quad_type="RADAU-RIGHT"):
    if odt_max is None:
        odt_max = 20.0 if component == "implicit" else 5.0
    odt_vals = np.linspace(0, odt_max, n + 1)[1:]

    groups = IMPLICIT_GROUPS if component == "implicit" else EXPLICIT_GROUPS

    for group_name, cfg in groups:
        if cfg["scheme_type"] == "rk":
            curves = rk_curves(odt_vals, component)
        else:
            curves = sdc_curves(
                odt_vals, component,
                qdelta_imp_type=cfg["qdelta_imp_type"],
                qdelta_exp_type=cfg["qdelta_exp_type"],
                node_type=node_type, quad_type=quad_type,
                quad_update=quad_update,
            )

        comp_label = (
            r"Implicit ($z_I = i\omega\Delta t$, $z_E=0$)"
            if component == "implicit"
            else r"Explicit ($z_E = i\omega\Delta t$, $z_I=0$)"
        )
        title = f"Phase Error — {group_name} — {comp_label}"
        fig = plot_phase_group(odt_vals, curves, title, odt_max)
        fname = f"phase_error_{component}_{group_name.lower().replace('-','_')}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {fname}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["implicit", "explicit", "both"], default="both")
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--odt-max-imp", type=float, default=20.0)
    parser.add_argument("--odt-max-exp", type=float, default=5.0)
    parser.add_argument("--no-quad-update", dest="quad_update", action="store_false", default=True)
    parser.add_argument("--node-type", default="LEGENDRE")
    parser.add_argument("--quad-type", default="RADAU-RIGHT")
    args = parser.parse_args()

    if args.component in ("implicit", "both"):
        run("implicit", n=args.n, odt_max=args.odt_max_imp,
            quad_update=args.quad_update,
            node_type=args.node_type, quad_type=args.quad_type)
    if args.component in ("explicit", "both"):
        run("explicit", n=args.n, odt_max=args.odt_max_exp,
            quad_update=args.quad_update,
            node_type=args.node_type, quad_type=args.quad_type)