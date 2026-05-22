"""Run all stability plots for both wave and Dahlquist test equations.

Configurations:
  SDC MIN-SR-S   + PIC  RADAU-RIGHT  no quad update
  SDC MIN-SR-FLEX + PIC RADAU-RIGHT  no quad update
  RK  (all variants)

To add IDC / RIDC later, append entries to SDC_CONFIGS or add dedicated
plot functions following the same pattern.
"""

import argparse

from plot_dahlquist_stability import plot_component_family_rk, plot_component_family_sdc
from plot_wave_stability import plot_wave_rk, plot_wave_sdc

# ── SDC configurations to sweep over ────────────────────────────────────────
SDC_CONFIGS = [
    dict(
        tag="MIN-SR-S_PIC",
        qdelta_imp_type="MIN-SR-S",
        qdelta_exp_type="PIC",
        node_type="LEGENDRE",
        quad_type="RADAU-RIGHT",
        quad_update=False,
    ),
    dict(
        tag="MIN-SR-FLEX_PIC",
        qdelta_imp_type="MIN-SR-FLEX",
        qdelta_exp_type="PIC",
        node_type="LEGENDRE",
        quad_type="RADAU-RIGHT",
        quad_update=False,
    ),
    dict(
        tag="LU_FE",
        qdelta_imp_type="LU",
        qdelta_exp_type="PIC",
        node_type="LEGENDRE",
        quad_type="RADAU-RIGHT",
        quad_update=False,
    ),
    dict(
        tag="BE_FE",
        qdelta_imp_type="BE",
        qdelta_exp_type="FE",
        node_type="LEGENDRE",
        quad_type="RADAU-RIGHT",
        quad_update=True,
    ),
    dict(
        tag="BE_FE_EQUID_LOBATTO",
        qdelta_imp_type="BE",
        qdelta_exp_type="FE",
        node_type="EQUID",
        quad_type="LOBATTO",
        quad_update=False,
    ),
    # ── IDC / RIDC entries go here ──────────────────────────────────────────
]


def run_wave(n=300):
    print("\n=== Wave stability ===")
    for cfg in SDC_CONFIGS:
        tag = cfg["tag"]
        out = f"stability_wave_sdc_{tag}.png"
        print(f"  SDC  {tag} -> {out}")
        plot_wave_sdc(
            n=n,
            output=out,
            quad_update=cfg["quad_update"],
            qdelta_imp_type=cfg["qdelta_imp_type"],
            qdelta_exp_type=cfg["qdelta_exp_type"],
            node_type=cfg["node_type"],
            quad_type=cfg["quad_type"],
        )

    print("  RK  (all) -> stability_wave_rk.png")
    plot_wave_rk(n=n, output="stability_wave_rk.png")


def run_dahlquist(n=160):
    print("\n=== Dahlquist stability ===")
    seen_exp_keys = set()
    for cfg in SDC_CONFIGS:
        # Explicit component only depends on qdelta_exp — skip duplicates.
        exp_key = (cfg["qdelta_exp_type"], cfg["node_type"], cfg["quad_type"])
        if exp_key not in seen_exp_keys:
            seen_exp_keys.add(exp_key)
            exp_tag = cfg["qdelta_exp_type"]
            out = f"stability_dahlquist_explicit_sdc_{exp_tag}.png"
            print(f"  SDC  explicit ({exp_tag}) -> {out}")
            plot_component_family_sdc(
                "explicit",
                n=n,
                output=out,
                quad_update=cfg["quad_update"],
                qdelta_imp_type=cfg["qdelta_imp_type"],
                qdelta_exp_type=cfg["qdelta_exp_type"],
                node_type=cfg["node_type"],
                quad_type=cfg["quad_type"],
            )
        else:
            print(f"  SDC  explicit ({cfg['qdelta_exp_type']}) already plotted — skipping")

        # Implicit component only depends on qdelta_imp — always unique per config.
        imp_tag = cfg["qdelta_imp_type"]
        out = f"stability_dahlquist_implicit_sdc_{imp_tag}.png"
        print(f"  SDC  implicit ({imp_tag}) -> {out}")
        plot_component_family_sdc(
            "implicit",
            n=n,
            output=out,
            quad_update=cfg["quad_update"],
            qdelta_imp_type=cfg["qdelta_imp_type"],
            qdelta_exp_type=cfg["qdelta_exp_type"],
            node_type=cfg["node_type"],
            quad_type=cfg["quad_type"],
        )

    for component in ("explicit", "implicit"):
        out = f"stability_dahlquist_{component}_rk.png"
        print(f"  RK  (all)  {component} -> {out}")
        plot_component_family_rk(component, n=n, output=out)


def main():
    parser = argparse.ArgumentParser(description="Run all stability plots.")
    parser.add_argument(
        "--problem",
        choices=["wave", "dahlquist", "all"],
        default="all",
        help="Which equation set to plot (default: all)",
    )
    parser.add_argument("--n-wave", type=int, default=300, help="Grid size for wave plots")
    parser.add_argument("--n-dahlquist", type=int, default=160, help="Grid size for Dahlquist plots")
    args = parser.parse_args()

    if args.problem in ("wave", "all"):
        run_wave(n=args.n_wave)
    if args.problem in ("dahlquist", "all"):
        run_dahlquist(n=args.n_dahlquist)

    print("\nDone.")


if __name__ == "__main__":
    main()
