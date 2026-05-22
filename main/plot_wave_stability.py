import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qmat import genQCoeffs, genQDeltaCoeffs
from stability_functions import (
    add_shared_colorbar,
    build_imex_rk_schemes,
    imex_rk_stability_grid,
    plot_wave_panel,
    sdc_stability_imex_grid,
)


DEFAULT_MK_LIST = [(2, [1, 2, 3, 4]), (3, [2, 3, 4, 5]), (4, [3, 4, 5, 6]), (5, [4, 5, 6, 7])]
#DEFAULT_MK_LIST = [(3, [2, 3, 4, 8, 15])]

STAB_TOL = 1.01  # values within this of 1.0 are floating point noise


def compute_sdc_qdelta_imp_seq(m, k, nodes, qmat, qdelta_imp_type, node_type, quad_type):
    if qdelta_imp_type == "MIN-SR-FLEX":
        return [
            genQDeltaCoeffs(
                "MIN-SR-FLEX",
                nNodes=m,
                nodeType=node_type,
                quadType=quad_type,
                nodes=nodes,
                Q=qmat,
                k=sweep,
            )
            for sweep in range(1, k + 1)
        ]
    return None


def apply_stab_tol(r_mod, tol=STAB_TOL):
    """
    Clip values in (1.0, tol] down to 0.9999 to suppress floating point noise
    near the stability boundary. Values genuinely above tol are left untouched
    and will still appear as unstable in the plot.
    """
    return np.where(r_mod <= tol, np.minimum(r_mod, 0.9999), r_mod)


def plot_wave_sdc(
    n=300,
    output="stability_wave_sdc.png",
    quad_update=True,
    qdelta_imp_type="LU",
    qdelta_exp_type="FE",
    node_type="LEGENDRE",
    quad_type="RADAU-RIGHT",
):
    fdt = np.linspace(0, 12, n)
    sdt = np.linspace(0, 5, n)

    mk_list = DEFAULT_MK_LIST
    n_rows = len(mk_list)
    n_cols = max(len(ks) for _, ks in mk_list)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)

    for row, (m, ks) in enumerate(mk_list):
        nodes, weights, qmat = genQCoeffs(
            "Collocation", nNodes=m, nodeType="LEGENDRE", quadType="RADAU-RIGHT"
        )
        qdelta_exp = genQDeltaCoeffs(qdelta_exp_type, nodes=nodes, Q=qmat)

        for col, k in enumerate(ks):
            qdelta_imp_seq = compute_sdc_qdelta_imp_seq(
                m, k, nodes, qmat, qdelta_imp_type, node_type, quad_type
            )
            qdelta_imp = None
            if qdelta_imp_seq is None:
                qdelta_imp = genQDeltaCoeffs(
                    qdelta_imp_type,
                    nNodes=m,
                    nodeType=node_type,
                    quadType=quad_type,
                    nodes=nodes,
                    Q=qmat,
                )

            dtf_grid = 1j * fdt[:, np.newaxis] * np.ones(n)
            dts_grid = 1j * np.ones(n)[:, np.newaxis] * sdt
            vals = sdc_stability_imex_grid(
                dtf_grid,
                dts_grid,
                nodes,
                weights,
                qmat,
                qdelta_imp,
                qdelta_exp,
                k,
                quad_update=quad_update,
                qdelta_imp_type=qdelta_imp_type if qdelta_imp_seq is not None else None,
                qdelta_imp_seq=qdelta_imp_seq,
            )
            r_mod = apply_stab_tol(np.abs(vals))

            plot_wave_panel(axes[row][col], sdt, fdt, r_mod, f"M={m}, K={k}")

        for col in range(len(ks), n_cols):
            axes[row][col].set_visible(False)

    fig.subplots_adjust(right=0.88, top=0.93)
    add_shared_colorbar(fig)
    fig.suptitle(
        f"IMEX Stability - SDC ({qdelta_imp_type} + {qdelta_exp_type})\n"
        + r"$dy/dt + is\,y + if\,y = 0$",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def plot_wave_rk(n=300, output="stability_wave_rk.png"):
    fdt = np.linspace(0, 12, n)
    sdt = np.linspace(0, 5, n)
    schemes = build_imex_rk_schemes()

    n_schemes = len(schemes)
    n_cols = 3
    n_rows = int(np.ceil(n_schemes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)

    for idx, (name, scheme) in enumerate(schemes.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        fdt_grid = fdt[:, np.newaxis] * np.ones(n)
        sdt_grid = np.ones(n)[:, np.newaxis] * sdt
        vals = imex_rk_stability_grid(
            fdt_grid, sdt_grid,
            scheme["A_imp"], scheme["A_exp"],
            scheme["b_imp"], scheme["b_exp"],
        )
        r_mod = apply_stab_tol(np.abs(vals))
        plot_wave_panel(ax, sdt, fdt, r_mod, name)

    for idx in range(n_schemes, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(right=0.88, top=0.93)
    add_shared_colorbar(fig)
    fig.suptitle("IMEX RK Stability\n" + r"$dy/dt + is\,y + if\,y = 0$", fontsize=13, fontweight="bold")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot wave-equation IMEX stability diagrams.")
    parser.add_argument("--scheme", choices=["SDC", "RK"], default="SDC")
    parser.add_argument("--n", type=int, default=300, help="Grid size per axis")
    parser.add_argument("--output", default=None, help="Output PNG path")
    parser.add_argument("--quad-update", dest="quad_update", action="store_true", default=True)
    parser.add_argument("--no-quad-update", dest="quad_update", action="store_false")
    parser.add_argument("--qdelta-imp-type", default="LU")
    parser.add_argument("--qdelta-exp-type", default="FE")
    parser.add_argument("--node-type", default="LEGENDRE")
    parser.add_argument("--quad-type", default="RADAU-RIGHT")
    args = parser.parse_args()

    if args.scheme == "RK":
        out = args.output or "stability_wave_rk.png"
        plot_wave_rk(n=args.n, output=out)
    else:
        out = args.output or "stability_wave_sdc.png"
        plot_wave_sdc(
            n=args.n,
            output=out,
            quad_update=args.quad_update,
            qdelta_imp_type=args.qdelta_imp_type,
            qdelta_exp_type=args.qdelta_exp_type,
            node_type=args.node_type,
            quad_type=args.quad_type,
        )


if __name__ == "__main__":
    main()