import numpy as np
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from qmat import genQDeltaCoeffs
from qmat.qcoeff.butcher import ARK548L2SAESDIRK2, ARK548L2SAERK2


def plot_wave_panel(ax, sdt, fdt, r_mod, title):
    r_plot = np.where(r_mod <= 1.0, r_mod, np.nan)
    ax.contourf(sdt, fdt, r_mod, levels=[1.0, r_mod.max() + 0.1], colors=["#d0d0d0"])
    cf = ax.contourf(
        sdt,
        fdt,
        r_plot,
        levels=np.linspace(0, 1, 50),
        cmap="viridis_r",
        vmin=0,
        vmax=1,
    )
    ax.contour(sdt, fdt, r_mod, levels=[1.0], colors="steelblue", linewidths=1.8)
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlim(sdt.min(), sdt.max())
    ax.set_ylim(fdt.min(), fdt.max())
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(r"$s\Delta t$  (slow/explicit)", fontsize=9)
    ax.set_ylabel(r"$f\Delta t$  (fast/implicit)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    return cf


def plot_complex_panel(ax, xvals, yvals, r_mod, title, xlabel=r"Re(z)", ylabel=r"Im(z)"):
    r_plot = np.where(r_mod <= 1.0, r_mod, np.nan)
    ax.contourf(xvals, yvals, r_mod, levels=[1.0, r_mod.max() + 0.1], colors=["#d0d0d0"])
    cf = ax.contourf(
        xvals,
        yvals,
        r_plot,
        levels=np.linspace(0, 1, 50),
        cmap="viridis_r",
        vmin=0,
        vmax=1,
    )
    ax.contour(xvals, yvals, r_mod, levels=[1.0], colors="steelblue", linewidths=1.8)
    ax.axvline(0, color="black", linewidth=0.6, linestyle=":")
    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlim(xvals.min(), xvals.max())
    ax.set_ylim(yvals.min(), yvals.max())
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    return cf


def add_shared_colorbar(fig, label=r"$|R|$ (stable region)"):
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap="viridis_r", norm=mcolors.Normalize(0, 1))
    fig.colorbar(sm, cax=cbar_ax, label=label)


def sdc_stability_imex(
    dtf,
    dts,
    nodes,
    weights,
    qmat,
    qdelta_imp,
    qdelta_exp,
    k,
    quad_update=True,
    qdelta_imp_type=None,
    node_type=None,
    quad_type=None,
    qdelta_imp_seq=None,
):
    m = len(nodes)
    ones = np.ones((m, 1))
    ident = np.eye(m)

    if qdelta_imp_type is None:
        lmat = ident - (dtf * qdelta_imp + dts * qdelta_exp)
        linv = np.linalg.inv(lmat)
        rmat = -dtf * qdelta_imp - dts * qdelta_exp + (dtf + dts) * qmat

        mpower = ident.copy()
        for _ in range(k):
            mpower = linv @ rmat @ mpower

        lpower = linv.copy()
        for i in range(1, k):
            mpower_i = ident.copy()
            for _ in range(i):
                mpower_i = linv @ rmat @ mpower_i
            lpower += mpower_i @ linv

        u1 = (mpower + lpower) @ ones
    else:
        linv_seq = []
        a_seq = []
        for sweep in range(k):
            if qdelta_imp_seq is not None:
                qdelta_imp_k = qdelta_imp_seq[sweep]
            else:
                qdelta_imp_k = genQDeltaCoeffs(
                    qdelta_imp_type,
                    nNodes=m,
                    nodeType=node_type,
                    quadType=quad_type,
                    nodes=nodes,
                    Q=qmat,
                    k=sweep + 1,
                )

            lmat = ident - (dtf * qdelta_imp_k + dts * qdelta_exp)
            rmat = -dtf * qdelta_imp_k - dts * qdelta_exp + (dtf + dts) * qmat
            linv_k = np.linalg.inv(lmat)
            linv_seq.append(linv_k)
            a_seq.append(linv_k @ rmat)

        mpower = ident.copy()
        for sweep in range(k):
            mpower = a_seq[sweep] @ mpower

        lpower = np.zeros((m, m), dtype=complex)
        suffix = ident.copy()
        for j in range(k - 1, -1, -1):
            lpower += suffix @ linv_seq[j]
            if j > 0:
                suffix = suffix @ a_seq[j]

        u1 = (mpower + lpower) @ ones

    sf = 1 + (dtf + dts) * (weights @ u1).item()
    return sf if quad_update else u1[-1][0]


def sdc_stability_imex_grid(
    dtf_grid,
    dts_grid,
    nodes,
    weights,
    qmat,
    qdelta_imp,
    qdelta_exp,
    k,
    quad_update=True,
    qdelta_imp_type=None,
    qdelta_imp_seq=None,
):
    """Vectorized version of sdc_stability_imex over a 2D grid.

    dtf_grid, dts_grid: shape (N1, N2) complex arrays (already scaled, e.g. 1j*f).
    Returns: shape (N1, N2) array of |R| values.

    Uses batched np.linalg.inv / matmul — no Python loop over grid points.
    """
    m = len(nodes)
    shape = dtf_grid.shape
    ident = np.eye(m, dtype=complex)

    # Add matrix dims for broadcasting: (..., M, M)
    dtf = dtf_grid[..., np.newaxis, np.newaxis]
    dts = dts_grid[..., np.newaxis, np.newaxis]
    qe = qdelta_exp[np.newaxis, np.newaxis]
    qm = qmat[np.newaxis, np.newaxis]
    id_batch = np.broadcast_to(ident, shape + (m, m)).copy()

    if qdelta_imp_type is None:
        qi = qdelta_imp[np.newaxis, np.newaxis]
        linv = np.linalg.inv(ident - dtf * qi - dts * qe)
        rmat = -dtf * qi - dts * qe + (dtf + dts) * qm
        a = linv @ rmat

        mpower = id_batch.copy()
        for _ in range(k):
            mpower = a @ mpower

        lpower = linv.copy()
        for i in range(1, k):
            ai = id_batch.copy()
            for _ in range(i):
                ai = a @ ai
            lpower = lpower + ai @ linv

        u1 = ((mpower + lpower) @ np.ones((m, 1)))[..., 0]
    else:
        linv_seq = []
        a_seq = []
        for sweep in range(k):
            qi_k = qdelta_imp_seq[sweep][np.newaxis, np.newaxis]
            linv_k = np.linalg.inv(ident - dtf * qi_k - dts * qe)
            rmat_k = -dtf * qi_k - dts * qe + (dtf + dts) * qm
            linv_seq.append(linv_k)
            a_seq.append(linv_k @ rmat_k)

        mpower = id_batch.copy()
        for sweep in range(k):
            mpower = a_seq[sweep] @ mpower

        lpower = np.zeros(shape + (m, m), dtype=complex)
        suffix = id_batch.copy()
        for j in range(k - 1, -1, -1):
            lpower = lpower + suffix @ linv_seq[j]
            if j > 0:
                suffix = suffix @ a_seq[j]

        u1 = ((mpower + lpower) @ np.ones((m, 1)))[..., 0]

    if quad_update:
        return 1.0 + (dtf_grid + dts_grid) * (u1 @ weights)
    else:
        return u1[..., -1]


def imex_rk_stability_grid(fdt_grid, sdt_grid, a_imp, a_exp, b_imp, b_exp):
    """Vectorized imex_rk_stability over a 2D grid.

    fdt_grid, sdt_grid: shape (N1, N2) real arrays.
    Returns: shape (N1, N2) complex array.
    """
    nu = len(b_imp)
    shape = fdt_grid.shape
    stages = np.zeros(shape + (nu,), dtype=complex)
    for j in range(nu):
        rhs = np.ones(shape, dtype=complex)
        for l in range(j):
            rhs -= 1j * sdt_grid * a_exp[j, l] * stages[..., l]
            rhs -= 1j * fdt_grid * a_imp[j, l] * stages[..., l]
        stages[..., j] = rhs / (1.0 + 1j * fdt_grid * a_imp[j, j])
    result = np.ones(shape, dtype=complex)
    for j in range(nu):
        result -= 1j * sdt_grid * b_exp[j] * stages[..., j]
        result -= 1j * fdt_grid * b_imp[j] * stages[..., j]
    return result


def imex_rk_stability(fdt, sdt, a_imp, a_exp, b_imp, b_exp):
    nu = len(b_imp)
    stages = np.zeros(nu, dtype=complex)
    for j in range(nu):
        rhs = 1.0
        for l in range(j):
            rhs -= 1j * sdt * a_exp[j, l] * stages[l]
            rhs -= 1j * fdt * a_imp[j, l] * stages[l]
        stages[j] = rhs / (1.0 + 1j * fdt * a_imp[j, j])

    result = 1.0
    for j in range(nu):
        result -= 1j * sdt * b_exp[j] * stages[j]
        result -= 1j * fdt * b_imp[j] * stages[j]
    return result


def build_imex_rk_schemes():
    schemes = {}

    schemes["IMEX Euler"] = dict(
        A_imp=np.array([[0.0, 0.0], [0.0, 1.0]]),
        A_exp=np.array([[0.0, 0.0], [1.0, 0.0]]),
        b_imp=np.array([0.0, 1.0]),
        b_exp=np.array([1.0, 0.0]),
    )

    g = (3.0 + np.sqrt(3.0)) / 6.0
    schemes["ARS3(2,3,3)"] = dict(
        A_imp=np.array([[0.0, 0.0, 0.0], [0.0, g, 0.0], [0.0, 1 - 2.0 * g, g]]),
        A_exp=np.array([[0.0, 0.0, 0.0], [g, 0.0, 0.0], [g - 1.0, 2.0 * (1.0 - g), 0.0]]),
        b_imp=np.array([0.0, 0.5, 0.5]),
        b_exp=np.array([0.0, 0.5, 0.5]),
    )

    schemes["ARS3(4,4,3)"] = dict(
        A_imp=np.array([[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 1/2, 0.0, 0.0, 0.0], [0.0, 1/6, 1/2, 0.0, 0.0], [0.0, -1/2, 1/2, 1/2, 0.0], [0.0, 3/2, -3/2, 1/2, 1/2]]),
        A_exp=np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.0], [11/18, 1/18, 0.0, 0.0, 0.0], [5/6, -5/6, 1/2, 0.0, 0.0], [1/4, 7/4, 3/4, -7/4, 0.0]]),
        b_imp=np.array([0, 3/2, -3/2,1/2, 1/2]),
        b_exp=np.array([1/4, 7/4, 3/4, -7/4, 0.0]),
    )


    g = 1.0 - 1.0 / np.sqrt(2.0)
    schemes["SSP3(3,3,2)"] = dict(
        A_imp=np.array([[g, 0.0, 0.0], [1 - 2.0 * g, g, 0.0], [0.5 - g, 0.0, g]]),
        A_exp=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 0.25, 0.0]]),
        b_imp=np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
        b_exp=np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]),
    )

    # g = 1767732205903 / 4055673282236
    # schemes["AR3(4)_L[2]SA"] = dict(
    #     A_imp=np.array(
    #         [
    #             [0.0, 0.0, 0.0, 0.0],
    #             [g, g, 0.0, 0.0],
    #             [2746238789719 / 10658868560708, -640167445237 / 6845629431997, g, 0.0],
    #             [
    #                 1471266399579 / 7840856788654,
    #                 -4482444167858 / 7529755066697,
    #                 11266239266428 / 11593286722821,
    #                 g,
    #             ],
    #         ]
    #     ),
    #     A_exp=np.array(
    #         [
    #             [0.0, 0.0, 0.0, 0.0],
    #             [1767732205903 / 2027836641118, 0.0, 0.0, 0.0],
    #             [5535828885825 / 10492691773637, 788022342437 / 10882634858940, 0.0, 0.0],
    #             [
    #                 6485989280629 / 16251701735622,
    #                 -4246266847089 / 9704473918619,
    #                 10755448449292 / 10357097424841,
    #                 0.0,
    #             ],
    #         ]
    #     ),
    #     b_imp=np.array(
    #         [
    #             1471266399579 / 7840856788654,
    #             -4482444167858 / 7529755066697,
    #             11266239266428 / 11593286722821,
    #             g,
    #         ]
    #     ),
    #     b_exp=np.array(
    #         [
    #             1471266399579 / 7840856788654,
    #             -4482444167858 / 7529755066697,
    #             11266239266428 / 11593286722821,
    #             1767732205903 / 4055673282236,
    #         ]
    #     ),
    # )

    schemes["ARK4(3)6L[2]SA"] = dict(
        A_imp=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 4, 1 / 4, 0, 0, 0, 0],
                [8611 / 62500, -1743 / 31250, 1 / 4, 0, 0, 0],
                [5012029 / 34652500, -654441 / 2922500, 174375 / 388108, 1 / 4, 0, 0],
                [
                    15267082809 / 155376265600,
                    -71443401 / 120774400,
                    730878875 / 902184768,
                    2285395 / 8070912,
                    1 / 4,
                    0,
                ],
                [82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, 1 / 4],
            ],
            dtype=float,
        ),
        A_exp=np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [1 / 2, 0, 0, 0, 0, 0],
                [13861 / 62500, 6889 / 62500, 0, 0, 0, 0],
                [
                    -116923316275 / 2393684061468,
                    -2731218467317 / 15368042101831,
                    9408046702089 / 11113171139209,
                    0,
                    0,
                    0,
                ],
                [
                    -451086348788 / 2902428689909,
                    -2682348792572 / 7519795681897,
                    12662868775082 / 11960479115383,
                    3355817975965 / 11060851509271,
                    0,
                    0,
                ],
                [
                    647845179188 / 3216320057751,
                    73281519250 / 8382639484533,
                    552539513391 / 3454668386233,
                    3354512671639 / 8306763924573,
                    4040 / 17871,
                    0,
                ],
            ],
            dtype=float,
        ),
        b_imp=np.array([82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, 1 / 4]),
        b_exp=np.array([82889 / 524892, 0, 15625 / 83664, 69875 / 102672, -2260 / 8211, 1 / 4]),
    )

    dirk = ARK548L2SAESDIRK2()
    erk = ARK548L2SAERK2()
    schemes["ARK5(4)8L[2]SA"] = dict(A_imp=dirk.A, A_exp=erk.A, b_imp=dirk.b, b_exp=erk.b)

    return schemes
