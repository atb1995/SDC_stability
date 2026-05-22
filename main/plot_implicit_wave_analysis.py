"""
plot_implicit_wave_analysis.py

Recreates the SI vs TR-BDF2 style plot (damping factor + phase shift)
for the implicit components of all schemes, one figure per order.

Left panel:  1 - |A|^2  (damping factor, want ~0 for slow modes, large for fast)
Right panel: phase shift theta = arctan2(Im(A), Re(A)) in radians (want ~0 for slow)

x-axis: omega*dt on log scale, covering Rossby through acoustic wave regimes.

Physical wave regime markers based on dt=240s, dx=10km:
  Rossby:   omega*dt ~ 0.02
  Gravity:  omega*dt ~ 1.5
  Acoustic: omega*dt ~ 8
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmat import genQCoeffs, genQDeltaCoeffs
from stability_functions import build_imex_rk_schemes, sdc_stability_imex_grid


# ---------------------------------------------------------------------------
# Wave regime markers (omega*dt, label, colour)
# ---------------------------------------------------------------------------
WAVE_REGIMES = [
    (0.02, "Rossby",   "#e377c2"),
    (1.5,  "Gravity",  "#2ca02c"),
    (8.0,  "Acoustic", "#ff7f0e"),
]

# ---------------------------------------------------------------------------
# Scheme definitions per order
# Each entry: (label, type, params)
# type = 'rk' -> params is scheme name string
# type = 'sdc' -> params is dict(m, k, qdelta_imp_type, quad_update)
# ---------------------------------------------------------------------------
ORDERS = {
    2: [
        ("IMEX Euler",             "rk",  "IMEX Euler"),
        ("SSP3(3,3,2)",            "rk",  "SSP3(3,3,2)"),
        ("SDC LU M=2 K=2",         "sdc", dict(m=2, k=2, qdelta_imp_type="LU",          quad_update=True)),
        ("SDC MIN-SR-S M=2 K=2",   "sdc", dict(m=2, k=2, qdelta_imp_type="MIN-SR-S",    quad_update=False)),
        ("SDC MIN-SR-FLEX M=2 K=2","sdc", dict(m=2, k=2, qdelta_imp_type="MIN-SR-FLEX", quad_update=False)),
    ],
    3: [
        ("ARS3(2,3,3)",            "rk",  "ARS3(2,3,3)"),
        ("ARS3(4,4,3)",            "rk",  "ARS3(4,4,3)"),
        ("SDC LU M=2 K=3",         "sdc", dict(m=2, k=3, qdelta_imp_type="LU",          quad_update=True)),
        ("SDC MIN-SR-S M=2 K=3",   "sdc", dict(m=2, k=3, qdelta_imp_type="MIN-SR-S",    quad_update=False)),
        ("SDC MIN-SR-FLEX M=2 K=3","sdc", dict(m=2, k=3, qdelta_imp_type="MIN-SR-FLEX", quad_update=False)),
    ],
    4: [
        ("ARK4(3)6L[2]SA",         "rk",  "ARK4(3)6L[2]SA"),
        ("SDC LU M=3 K=4",         "sdc", dict(m=3, k=4, qdelta_imp_type="LU",          quad_update=True)),
        ("SDC MIN-SR-S M=3 K=4",   "sdc", dict(m=3, k=4, qdelta_imp_type="MIN-SR-S",    quad_update=False)),
        ("SDC MIN-SR-FLEX M=3 K=4","sdc", dict(m=3, k=4, qdelta_imp_type="MIN-SR-FLEX", quad_update=False)),
    ],
    5: [
        ("ARK5(4)8L[2]SA",         "rk",  "ARK5(4)8L[2]SA"),
        ("SDC LU M=3 K=5",         "sdc", dict(m=3, k=5, qdelta_imp_type="LU",          quad_update=True)),
        ("SDC MIN-SR-S M=3 K=5",   "sdc", dict(m=3, k=5, qdelta_imp_type="MIN-SR-S",    quad_update=False)),
    ],
}

NODE_TYPE = "LEGENDRE"
QUAD_TYPE = "RADAU-RIGHT"
N_POINTS  = 600


# ---------------------------------------------------------------------------
# Amplification factor computation
# ---------------------------------------------------------------------------

def _rk_implicit_A(odt_vals, scheme):
    """Implicit-only amplification for an RK scheme: z_E=0, z_I=i*odt."""
    from plot_dahlquist_stability import _imex_rk_stability_complex_grid
    z = 1j * odt_vals
    zero = np.zeros_like(z)
    return _imex_rk_stability_complex_grid(
        z[np.newaxis, :], zero[np.newaxis, :],
        scheme["A_imp"], scheme["A_exp"],
        scheme["b_imp"], scheme["b_exp"],
    )[0]


def _sdc_implicit_A(odt_vals, m, k, qdelta_imp_type, quad_update):
    """Implicit-only amplification for an SDC scheme: z_E=0, z_I=i*odt."""
    nodes, weights, qmat = genQCoeffs(
        "Collocation", nNodes=m, nodeType=NODE_TYPE, quadType=QUAD_TYPE)
    qdelta_exp = genQDeltaCoeffs("PIC", nodes=nodes, Q=qmat)

    qdelta_imp_seq = None
    qdelta_imp = None
    imp_type_arg = None

    if qdelta_imp_type == "MIN-SR-FLEX":
        qdelta_imp_seq = [
            genQDeltaCoeffs(
                "MIN-SR-FLEX", nNodes=m, nodeType=NODE_TYPE,
                quadType=QUAD_TYPE, nodes=nodes, Q=qmat, k=sweep,
            )
            for sweep in range(1, k + 1)
        ]
        imp_type_arg = "MIN-SR-FLEX"
    else:
        qdelta_imp = genQDeltaCoeffs(
            qdelta_imp_type, nNodes=m, nodeType=NODE_TYPE,
            quadType=QUAD_TYPE, nodes=nodes, Q=qmat,
        )

    z = 1j * odt_vals
    zero = np.zeros_like(z)

    return sdc_stability_imex_grid(
        z[np.newaxis, :], zero[np.newaxis, :],
        nodes, weights, qmat,
        qdelta_imp, qdelta_exp, k,
        quad_update=quad_update,
        qdelta_imp_type=imp_type_arg,
        qdelta_imp_seq=qdelta_imp_seq,
    )[0]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLORS = plt.cm.tab10.colors


def plot_order(order, schemes, rk_schemes, odt_vals):
    fig, (ax_damp, ax_phase) = plt.subplots(1, 2, figsize=(12, 5))

    for i, (label, stype, sparams) in enumerate(schemes):
        color = COLORS[i % len(COLORS)]

        if stype == "rk":
            A = _rk_implicit_A(odt_vals, rk_schemes[sparams])
        else:
            A = _sdc_implicit_A(odt_vals, **sparams)

        absA = np.abs(A)
        damp = 1.0 - absA**2
        phase = -np.unwrap(np.arctan2(np.imag(A), np.real(A)))  # negated to match convention

        ax_damp.semilogx(odt_vals, damp, linewidth=1.5, label=label, color=color)
        ax_phase.semilogx(odt_vals, phase, linewidth=1.5, label=label, color=color)

    # Wave regime markers
    for odt, wlabel, wcolor in WAVE_REGIMES:
        for ax in (ax_damp, ax_phase):
            ax.axvline(odt, color=wcolor, linewidth=1.0, linestyle="--", alpha=0.8)
        ax_damp.text(odt, 1.02, wlabel, color=wcolor, fontsize=7,
                     ha='center', va='bottom', transform=ax_damp.get_xaxis_transform())
        ax_phase.text(odt, 1.02, wlabel, color=wcolor, fontsize=7,
                      ha='center', va='bottom', transform=ax_phase.get_xaxis_transform())

    # Reference lines
    ax_damp.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax_damp.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
    ax_phase.axhline(0.0, color="black", linewidth=0.6, linestyle=":")

    ax_damp.set_xlabel(r"$\omega \Delta t$", fontsize=11)
    ax_damp.set_ylabel(r"$1 - |A|^2$  (damping factor)", fontsize=11)
    ax_damp.set_title("Damping factor", fontsize=12, fontweight="bold")
    ax_damp.set_ylim(-0.05, 1.1)
    ax_damp.set_xlim(odt_vals[0], odt_vals[-1])
    ax_damp.legend(fontsize=8, loc="upper left")
    ax_damp.grid(True, which="both", linewidth=0.3, alpha=0.4)

    ax_phase.set_xlabel(r"$\omega \Delta t$", fontsize=11)
    ax_phase.set_ylabel(r"Phase shift (radians)", fontsize=11)
    ax_phase.set_title("Phase shift", fontsize=12, fontweight="bold")
    ax_phase.set_xlim(odt_vals[0], odt_vals[-1])
    ax_phase.legend(fontsize=8, loc="lower left")
    ax_phase.grid(True, which="both", linewidth=0.3, alpha=0.4)

    fig.suptitle(
        f"Implicit component — Order {order}  "
        r"($z_E = 0$, $z_I = i\omega\Delta t$)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()

    fname = f"implicit_wave_analysis_order{order}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    odt_vals = np.logspace(-3, 2, N_POINTS)
    rk_schemes = build_imex_rk_schemes()

    for order, schemes in ORDERS.items():
        plot_order(order, schemes, rk_schemes, odt_vals)