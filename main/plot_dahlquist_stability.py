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
    plot_complex_panel,
    sdc_stability_imex_grid,
)


DEFAULT_MK_LIST = [(2, [1, 2, 3]), (3, [2, 3, 4]), (4, [3, 4, 5]), (5, [4, 5, 6])]


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


def compute_component_grid_sdc(
    n,
    xvals,
    yvals,
    nodes,
    weights,
    qmat,
    qdelta_imp,
    qdelta_exp,
    qdelta_imp_seq,
    k,
    component,
    quad_update,
    qdelta_imp_type,
    node_type,
    quad_type,
):
    # z_grid[i, j] = xvals[j] + 1j*yvals[i]
    z_grid = xvals[np.newaxis, :] + 1j * yvals[:, np.newaxis]  # (N, N)
    zero_grid = np.zeros_like(z_grid)
    if component == "explicit":
        dtf_grid, dts_grid = zero_grid, z_grid
    else:
        dtf_grid, dts_grid = z_grid, zero_grid
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
    return np.abs(vals)


def compute_component_grid_rk(n, xvals, yvals, scheme, component):
    # z_grid[i, j] = xvals[j] + 1j*yvals[i]
    z_grid = xvals[np.newaxis, :] + 1j * yvals[:, np.newaxis]  # (N, N)
    zero_grid = np.zeros_like(z_grid)
    if component == "explicit":
        # explicit arg maps to sdt in imex_rk_stability; imp arg = 0
        fdt_grid, sdt_grid = zero_grid.real, z_grid.real
        fdt_grid_i, sdt_grid_i = zero_grid.imag, z_grid.imag
        # z_grid is complex: Re+iIm; we pass it directly as 1j*z_imp / 1j*z_exp
        # simpler: use the complex grid directly
        z_imp = zero_grid
        z_exp = z_grid
    else:
        z_imp = z_grid
        z_exp = zero_grid
    # imex_rk_stability_grid expects real fdt, sdt but the Dahlquist plane is complex z.
    # We reuse the RK formula with fdt=Re(z_imp), sdt=Re(z_exp) and imaginary parts
    # by passing z_imp and z_exp as the full complex arguments to a helper that
    # accepts complex grids directly.
    vals = _imex_rk_stability_complex_grid(
        z_imp, z_exp, scheme["A_imp"], scheme["A_exp"], scheme["b_imp"], scheme["b_exp"]
    )
    return np.abs(vals)


def _imex_rk_stability_complex_grid(z_imp_grid, z_exp_grid, a_imp, a_exp, b_imp, b_exp):
    """RK stability over 2D complex grids z_imp, z_exp (Dahlquist test equation)."""
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


def plot_component_family_sdc(
    component,
    n=160,
    xlim=None,
    ylim=None,
    output=None,
    quad_update=True,
    qdelta_imp_type="LU",
    qdelta_exp_type="FE",
    node_type="LEGENDRE",
    quad_type="RADAU-RIGHT",
):
    if ylim is None:
        ylim = (-5.0, 5.0) if component == "explicit" else (-25.0, 25.0)
    if xlim is None:
        xlim = (-9.0, 1.0) if component == "explicit" else (-49.0, 1.0)
    xvals = np.linspace(xlim[0], xlim[1], n)
    yvals = np.linspace(ylim[0], ylim[1], n)

    mk_list = DEFAULT_MK_LIST
    n_rows = len(mk_list)
    n_cols = max(len(ks) for _, ks in mk_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)

    for row, (m, ks) in enumerate(mk_list):
        nodes, weights, qmat = genQCoeffs(
            "Collocation", nNodes=m, nodeType=node_type, quadType=quad_type
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

            r_mod = compute_component_grid_sdc(
                n,
                xvals,
                yvals,
                nodes,
                weights,
                qmat,
                qdelta_imp,
                qdelta_exp,
                qdelta_imp_seq,
                k,
                component,
                quad_update,
                qdelta_imp_type,
                node_type,
                quad_type,
            )
            if component == "explicit":
                label = "explicit"
            else:
                label = "implicit"
            plot_complex_panel(axes[row][col], xvals, yvals, r_mod, f"M={m}, K={k} ({label})")

        for col in range(len(ks), n_cols):
            axes[row][col].set_visible(False)

    fig.subplots_adjust(right=0.88, top=0.93)
    add_shared_colorbar(fig)

    if component == "explicit":
        variant = qdelta_exp_type
        component_title = r"Explicit component ($z_E$), with $z_I = 0$"
        out = output or "stability_dahlquist_explicit_sdc.png"
    else:
        variant = qdelta_imp_type
        component_title = r"Implicit component ($z_I$), with $z_E = 0$"
        out = output or "stability_dahlquist_implicit_sdc.png"

    fig.suptitle(
        f"IMEX Stability - SDC ({variant})\\n" + component_title,
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_component_family_rk(component, n=160, xlim=None, ylim=None, output=None):
    if ylim is None:
        ylim = (-5.0, 5.0) if component == "explicit" else (-25.0, 25.0)
    if xlim is None:
        xlim = (-9.0, 1.0) if component == "explicit" else (-49.0, 1.0)
    xvals = np.linspace(xlim[0], xlim[1], n)
    yvals = np.linspace(ylim[0], ylim[1], n)
    schemes = build_imex_rk_schemes()

    n_schemes = len(schemes)
    n_cols = 3
    n_rows = int(np.ceil(n_schemes / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.0 * n_rows), squeeze=False)

    for idx, (name, scheme) in enumerate(schemes.items()):
        row, col = divmod(idx, n_cols)
        r_mod = compute_component_grid_rk(n, xvals, yvals, scheme, component)
        plot_complex_panel(axes[row][col], xvals, yvals, r_mod, name)

    for idx in range(n_schemes, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.subplots_adjust(right=0.88, top=0.93)
    add_shared_colorbar(fig)

    if component == "explicit":
        component_title = r"Explicit component ($z_E$), with $z_I = 0$"
        out = output or "stability_dahlquist_explicit_rk.png"
    else:
        component_title = r"Implicit component ($z_I$), with $z_E = 0$"
        out = output or "stability_dahlquist_implicit_rk.png"

    fig.suptitle("IMEX RK Stability\\n" + component_title, fontsize=13, fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Dahlquist stability for explicit and implicit IMEX components separately."
    )
    parser.add_argument("--scheme", choices=["SDC", "RK"], default="SDC")
    parser.add_argument("--n", type=int, default=160, help="Grid size per axis")
    parser.add_argument(
        "--component",
        choices=["explicit", "implicit", "both"],
        default="both",
        help="Which component plane(s) to plot",
    )
    parser.add_argument("--quad-update", dest="quad_update", action="store_true", default=True)
    parser.add_argument("--no-quad-update", dest="quad_update", action="store_false")
    parser.add_argument("--qdelta-imp-type", default="LU")
    parser.add_argument("--qdelta-exp-type", default="FE")
    parser.add_argument("--node-type", default="LEGENDRE")
    parser.add_argument("--quad-type", default="RADAU-RIGHT")
    parser.add_argument("--xmin", type=float, default=None)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--ymax", type=float, default=None)
    parser.add_argument("--output", default=None, help="Output PNG path; valid for single-component runs")
    args = parser.parse_args()

    xlim = (args.xmin, args.xmax) if args.xmin is not None else None
    ylim = (args.ymin, args.ymax) if args.ymin is not None else None

    if args.component == "both" and args.output is not None:
        raise ValueError("--output can only be used with --component explicit or --component implicit")

    if args.component in ("explicit", "both"):
        if args.scheme == "RK":
            plot_component_family_rk("explicit", n=args.n, xlim=xlim, ylim=ylim, output=args.output)
        else:
            plot_component_family_sdc(
                "explicit",
                n=args.n,
                xlim=xlim,
                ylim=ylim,
                output=args.output,
                quad_update=args.quad_update,
                qdelta_imp_type=args.qdelta_imp_type,
                qdelta_exp_type=args.qdelta_exp_type,
                node_type=args.node_type,
                quad_type=args.quad_type,
            )
    if args.component in ("implicit", "both"):
        if args.scheme == "RK":
            plot_component_family_rk("implicit", n=args.n, xlim=xlim, ylim=ylim, output=args.output)
        else:
            plot_component_family_sdc(
                "implicit",
                n=args.n,
                xlim=xlim,
                ylim=ylim,
                output=args.output,
                quad_update=args.quad_update,
                qdelta_imp_type=args.qdelta_imp_type,
                qdelta_exp_type=args.qdelta_exp_type,
                node_type=args.node_type,
                quad_type=args.quad_type,
            )


if __name__ == "__main__":
    main()
