"""
check_sdt0.py

Sanity check: evaluate |R| along s*dt = 0 (pure implicit limit).
At sdt=0 we should exactly recover the implicit Dahlquist stability.
Any |R| > 1 on the imaginary axis (f*dt = i*f) is a bug.
"""

import numpy as np
from qmat import genQCoeffs, genQDeltaCoeffs
from stability_functions import sdc_stability_imex_grid

DEFAULT_MK_LIST = [(2, [1, 2, 3]), (3, [2, 3, 4]), (4, [3, 4, 5]), (5, [4, 5, 6])]

SCHEMES = [
    ("LU",          "FE",  False),
    ("BE",          "FE",  False),
    ("MIN-SR-FLEX", "PIC", True),
    ("MIN-SR-S",    "PIC", False),
]

fdt_vals = np.linspace(0, 12, 500)[1:]  # imaginary axis, exclude 0
node_type = "LEGENDRE"
quad_type = "RADAU-RIGHT"

for qdelta_imp_type, qdelta_exp_type, is_flex in SCHEMES:
    print(f"\n{'='*60}")
    print(f"Scheme: {qdelta_imp_type} + {qdelta_exp_type}")
    print(f"{'='*60}")
    any_unstable = False

    for m, ks in DEFAULT_MK_LIST:
        nodes, weights, qmat = genQCoeffs(
            "Collocation", nNodes=m, nodeType=node_type, quadType=quad_type
        )
        qdelta_exp = genQDeltaCoeffs(qdelta_exp_type, nodes=nodes, Q=qmat)

        for k in ks:
            qdelta_imp_seq = None
            qdelta_imp = None
            if is_flex:
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

            # sdt = 0, fdt = i*fdt_vals (purely imaginary)
            # dtf_grid = 1j * fdt_vals[np.newaxis, :]
            # dts_grid = np.zeros((1, len(fdt_vals)), dtype=complex)

            dts_grid = 1j * 0.1 * np.ones((1, len(fdt_vals)))
            dtf_grid = 1j * fdt_vals[np.newaxis, :]

            R = sdc_stability_imex_grid(
                dtf_grid, dts_grid,
                nodes, weights, qmat,
                qdelta_imp, qdelta_exp, k,
                quad_update=True,
                qdelta_imp_type=qdelta_imp_type if is_flex else None,
                qdelta_imp_seq=qdelta_imp_seq,
            )[0]

            

            absR = np.abs(R)
            max_absR = np.max(absR)
            unstable_fdt = fdt_vals[absR > 1.0]

            # unstable_vals = absR[absR > 1.0]
            # print(f"  |R| distribution above 1: min={unstable_vals.min():.6f}, "
            #     f"max={unstable_vals.max():.6f}, "
            #     f"median={np.median(unstable_vals):.6f}, "
            #     f">1.01: {(unstable_vals > 1.01).sum()}")

            # After computing R, for the first unstable case:
            # if len(unstable_fdt) > 0:
            #     idx = np.argmax(absR > 1.0)
            #     print(f"  At fdt={fdt_vals[idx]:.3f}: R={R[idx]:.6f}, |R|={absR[idx]:.6f}")
            #     # Also check condition number of L at that point
            #     dtf_val = 1j * fdt_vals[idx]
            #     lmat = np.eye(m) - dtf_val * qdelta_imp
            #     print(f"  Condition number of L: {np.linalg.cond(lmat):.2e}")

            if len(unstable_fdt) > 0:
                any_unstable = True
                print(f"  M={m} K={k}: BUG — unstable at sdt=0, max|R|={max_absR:.4f}, "
                      f"first unstable fdt={unstable_fdt[0]:.3f}")
            else:
                print(f"  M={m} K={k}: correct at sdt=0 — max|R|={max_absR:.6f}")

    if not any_unstable:
        print("  All (M,K) stable at sdt=0 ✓")