"""Unified stability plotting entry point.

Examples:
    python plot_stability.py wave --scheme SDC
    python plot_stability.py wave --scheme RK
    python plot_stability.py dahlquist --scheme SDC --component both
    python plot_stability.py dahlquist --scheme RK --component explicit
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot wave or Dahlquist IMEX stability diagrams.")
    parser.add_argument("problem", choices=["wave", "dahlquist"], help="Which test problem to plot")
    parser.add_argument("--scheme", choices=["SDC", "RK"], default="SDC")
    parser.add_argument("--n", type=int, default=300, help="Grid size per axis")
    parser.add_argument("--output", default=None, help="Output PNG path")

    # SDC options (defaults requested by user)
    parser.add_argument("--quad-update", dest="quad_update", action="store_true", default=True)
    parser.add_argument("--no-quad-update", dest="quad_update", action="store_false")
    parser.add_argument("--qdelta-imp-type", default="LU")
    parser.add_argument("--qdelta-exp-type", default="FE")
    parser.add_argument("--node-type", default="LEGENDRE")
    parser.add_argument("--quad-type", default="RADAU-RIGHT")

    # Dahlquist-only options
    parser.add_argument(
        "--component",
        choices=["explicit", "implicit", "both"],
        default="both",
        help="Dahlquist component to plot (ignored for wave)",
    )
    parser.add_argument("--xmin", type=float, default=-8.0)
    parser.add_argument("--xmax", type=float, default=2.0)
    parser.add_argument("--ymin", type=float, default=-8.0)
    parser.add_argument("--ymax", type=float, default=8.0)
    args = parser.parse_args()

    if args.problem == "wave":
        from plot_wave_stability import plot_wave_rk, plot_wave_sdc

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
        return

    xlim = (args.xmin, args.xmax)
    ylim = (args.ymin, args.ymax)

    if args.component == "both" and args.output is not None:
        raise ValueError("--output can only be used with --component explicit or --component implicit")

    from plot_dahlquist_stability import plot_component_family_rk, plot_component_family_sdc

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
