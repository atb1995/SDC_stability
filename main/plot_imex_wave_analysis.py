"""
plot_imex_implicit_analysis.py

For each order, plot damping (1-|A|^2) and phase shift (theta) of the
combined IMEX amplification factor A(i*s_fixed*dt, i*f*dt) vs f*dt,
with s*dt fixed at a representative advective/Rossby value.

This shows how each scheme damps and phase-shifts the fast implicit modes
in the presence of a slow explicit advective mode.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmat import genQCoeffs, genQDeltaCoeffs
from stability_functions import build_imex_rk_schemes, sdc_stability_imex_grid


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SDT_FIXED = 0.05   # fixed advective s*dt
N_POINTS  = 600
NODE_TYPE = "LEGENDRE"
QUAD_TYPE = "RADAU-RIGHT"

WAVE_REGIMES = [
    (0.02, "Rossby",   "#e377c2"),
    (1.5, "Gravity",  "#2ca02c"),
    (8.0, "Acoustic", "#ff7f0e"),
]

# ---------------------------------------------------------------------------
# Scheme definitions per order
# Each entry: (label, type, params)
# ---------------------------------------------------------------------------
ORDERS = {
    2: [
        ("IMEX Euler",              "rk",  "IMEX Euler"),
        ("SSP3(3,3,2)",             "rk",  "SSP3(3,3,2)"),
        ("SDC LU M=2 K=2",          "sdc", dict(m=2, k=2, qdelta_imp_type="LU",          qdelta_exp_type="FE",  node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=True)),
        ("SDC MIN-SR-S M=2 K=2",    "sdc", dict(m=2, k=2, qdelta_imp_type="MIN-SR-S",    qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("SDC MIN-SR-FLEX M=2 K=2", "sdc", dict(m=2, k=2, qdelta_imp_type="MIN-SR-FLEX", qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("IDC BE M=2 K=2",          "sdc", dict(m=2, k=2, qdelta_imp_type="BE",          qdelta_exp_type="FE",  node_type="EQUID",    quad_type="LOBATTO",     quad_update=False)),
    ],
    3: [
        ("ARS3(2,3,3)",             "rk",  "ARS3(2,3,3)"),
        ("ARS3(4,4,3)",             "rk",  "ARS3(4,4,3)"),
        ("SDC LU M=2 K=3",          "sdc", dict(m=2, k=3, qdelta_imp_type="LU",          qdelta_exp_type="FE",  node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=True)),
        ("SDC MIN-SR-S M=2 K=3",    "sdc", dict(m=2, k=3, qdelta_imp_type="MIN-SR-S",    qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("SDC MIN-SR-FLEX M=3 K=3", "sdc", dict(m=3, k=3, qdelta_imp_type="MIN-SR-FLEX", qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("IDC BE M=3 K=3",          "sdc", dict(m=3, k=3, qdelta_imp_type="BE",          qdelta_exp_type="FE",  node_type="EQUID",    quad_type="LOBATTO",     quad_update=False)),
    ],
    4: [
        ("ARK4(3)6L[2]SA",          "rk",  "ARK4(3)6L[2]SA"),
        ("SDC LU M=3 K=4",          "sdc", dict(m=3, k=4, qdelta_imp_type="LU",          qdelta_exp_type="FE",  node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=True)),
        ("SDC MIN-SR-S M=3 K=4",    "sdc", dict(m=3, k=4, qdelta_imp_type="MIN-SR-S",    qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("SDC MIN-SR-FLEX M=4 K=4", "sdc", dict(m=4, k=4, qdelta_imp_type="MIN-SR-FLEX", qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("IDC BE M=4 K=4",          "sdc", dict(m=4, k=4, qdelta_imp_type="BE",          qdelta_exp_type="FE",  node_type="EQUID",    quad_type="LOBATTO",     quad_update=False)),
    ],
    5: [
        ("ARK5(4)8L[2]SA",          "rk",  "ARK5(4)8L[2]SA"),
        ("SDC LU M=3 K=5",          "sdc", dict(m=3, k=5, qdelta_imp_type="LU",          qdelta_exp_type="FE",  node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=True)),
        ("SDC MIN-SR-S M=3 K=5",    "sdc", dict(m=3, k=5, qdelta_imp_type="MIN-SR-S",    qdelta_exp_type="PIC", node_type="LEGENDRE", quad_type="RADAU-RIGHT", quad_update=False)),
        ("IDC BE M=5 K=5",          "sdc", dict(m=5, k=5, qdelta_imp_type="BE",          qdelta_exp_type="FE",  node_type="EQUID",    quad_type="LOBATTO",     quad_update=False)),
    ],
}

COLORS = plt.cm.tab10.colors
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (1, 1))]


# ---------------------------------------------------------------------------
# Amplification factor
# ---------------------------------------------------------------------------

def _rk_A(sdt_arr, fdt_vals, scheme):
    from plot_dahlquist_stability import _imex_rk_stability_complex_grid
    return _imex_rk_stability_complex_grid(
        (1j * fdt_vals)[np.newaxis, :],
        (1j * sdt_arr)[np.newaxis, :],
        scheme["A_imp"], scheme["A_exp"],
        scheme["b_imp"], scheme["b_exp"],
    )[0]


def _sdc_A(sdt_arr, fdt_vals, m, k, qdelta_imp_type, qdelta_exp_type,
           node_type, quad_type, quad_update):
    nodes, weights, qmat = genQCoeffs(
        "Collocation", nNodes=m, nodeType=node_type, quadType=quad_type)
    qdelta_exp = genQDeltaCoeffs(qdelta_exp_type, nodes=nodes, Q=qmat)

    qdelta_imp_seq = None
    qdelta_imp = None
    imp_type_arg = None

    if qdelta_imp_type == "MIN-SR-FLEX":
        qdelta_imp_seq = [
            genQDeltaCoeffs(
                "MIN-SR-FLEX", nNodes=m, nodeType=node_type,
                quadType=quad_type, nodes=nodes, Q=qmat, k=sweep,
            )
            for sweep in range(1, k + 1)
        ]
        imp_type_arg = "MIN-SR-FLEX"
    else:
        qdelta_imp = genQDeltaCoeffs(
            qdelta_imp_type, nNodes=m, nodeType=node_type,
            quadType=quad_type, nodes=nodes, Q=qmat,
        )

    return sdc_stability_imex_grid(
        (1j * fdt_vals)[np.newaxis, :],
        (1j * sdt_arr)[np.newaxis, :],
        nodes, weights, qmat,
        qdelta_imp, qdelta_exp, k,
        quad_update=quad_update,
        qdelta_imp_type=imp_type_arg,
        qdelta_imp_seq=qdelta_imp_seq,
    )[0]


def get_A(sdt_arr, fdt_vals, stype, sparams, rk_schemes):
    if stype == "rk":
        return _rk_A(sdt_arr, fdt_vals, rk_schemes[sparams])
    else:
        return _sdc_A(sdt_arr, fdt_vals, **sparams)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_order(order, schemes, rk_schemes):
    fdt_vals = np.logspace(-3, 2, N_POINTS)
    sdt_arr = SDT_FIXED * np.ones_like(fdt_vals)

    fig, (ax_damp, ax_phase) = plt.subplots(1, 2, figsize=(13, 5))

    for i, (label, stype, sparams) in enumerate(schemes):
        color = COLORS[i % len(COLORS)]
        ls = LINESTYLES[i % len(LINESTYLES)]

        A = get_A(sdt_arr, fdt_vals, stype, sparams, rk_schemes)
        absA = np.abs(A)
        damp = 1.0 - absA**2
        phase = -np.unwrap(np.arctan2(np.imag(A), np.real(A)))

        ax_damp.semilogx(fdt_vals, damp, linewidth=1.5, label=label, color=color, ls=ls)
        ax_phase.semilogx(fdt_vals, phase, linewidth=1.5, label=label, color=color, ls=ls)

    # Wave regime markers
    for odt, wlabel, wcolor in WAVE_REGIMES:
        for ax in (ax_damp, ax_phase):
            ax.axvline(odt, color=wcolor, linewidth=1.0, linestyle="--", alpha=0.8)
            ax.text(odt, 1.02, wlabel, color=wcolor, fontsize=7,
                    ha='center', va='bottom', transform=ax.get_xaxis_transform())

    # Reference lines
    ax_damp.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax_damp.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax_phase.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    ax_damp.set_xlabel(r"$f\Delta t$  (fast/implicit)", fontsize=11)
    ax_damp.set_ylabel(r"$1 - |A|^2$  (damping factor)", fontsize=11)
    ax_damp.set_title("Damping factor", fontsize=12, fontweight="bold")
    ax_damp.set_ylim(-0.05, 1.1)
    ax_damp.set_xlim(fdt_vals[0], fdt_vals[-1])
    ax_damp.legend(fontsize=7, loc="upper left")
    ax_damp.grid(True, which="both", linewidth=0.3, alpha=0.4)

    ax_phase.set_xlabel(r"$f\Delta t$  (fast/implicit)", fontsize=11)
    ax_phase.set_ylabel("Phase shift (radians)", fontsize=11)
    ax_phase.set_title("Phase shift", fontsize=12, fontweight="bold")
    ax_phase.set_xlim(fdt_vals[0], fdt_vals[-1])
    ax_phase.legend(fontsize=7, loc="lower left")
    ax_phase.grid(True, which="both", linewidth=0.3, alpha=0.4)

    fig.suptitle(
        f"Implicit damping in IMEX context — Order {order}  "
        rf"($s\Delta t = {SDT_FIXED}$ fixed)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = f"imex_implicit_analysis_order{order}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    fdt_vals = np.logspace(-3, 2, N_POINTS)
    rk_schemes = build_imex_rk_schemes()
    for order, schemes in ORDERS.items():
        plot_order(order, schemes, rk_schemes)