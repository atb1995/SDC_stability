
import numpy as np
import matplotlib.pyplot as plt
from qmat import genQCoeffs, genQDeltaCoeffs

def imex_euler_stability(z1, z2):
    """
    Stability function for IMEX Euler:
    R(z1, z2) = (1 + z2) / (1 - z1)
    """
    z1 = np.asarray(z1, dtype=np.complex128)
    z2 = np.asarray(z2, dtype=np.complex128)
    return (1 + z2) / (1 - z1)

def sdc_stability(dtf, dts, nodes, weights, Q, Qdelta_imp, Qdelta_exp, k):
    """
    Stability function for SDC methods:
    R(dtf, dts) = 1 + (dtf + dts) * (weights @ u1)
    where u1 is the result of the matrix-vector product after k sweeps.
    """
    M = len(nodes)
    ones = np.ones((M, 1))
    I = np.eye(M)
    L = I - (dtf*Qdelta_imp + dts*Qdelta_exp)
    Linv = np.linalg.inv(L)
    R = -dtf*Qdelta_imp - dts*Qdelta_exp + (dtf+dts)*Q

    # Propagate through k sweeps
    Mpower = I.copy()
    for i in range(k):
        Mpower = Linv @ R @ Mpower

    # Error term
    Lpower = Linv.copy()
    for i in range(1,k):
        Mpoweri = I.copy()
        for j in range(i):
            Mpoweri = Linv @ R @ Mpoweri
        Mpoweri = Mpoweri @ Linv
        Lpower += Mpoweri


    # Finaly update over quadrature
    u1 = (Mpower + Lpower) @ ones
    sf = 1 + (dtf + dts) * (weights @ u1).item()
    return sf

M = 3
k = 5

qType = "RADAU-RIGHT"
impqd = "IE"
expqd = "EE"
nType= "LEGENDRE"

# Coefficients or specific collocation method
nodes, weights, Q = genQCoeffs(
    "Collocation", nNodes=M, nodeType=nType, quadType=qType)

# QDelta matrix from Implicit-Euler based SDC
QDelta_imp = genQDeltaCoeffs(impqd, nodes=nodes, Q=Q)
QDelta_exp = genQDeltaCoeffs(expqd, nodes=nodes)

# Grid
x = np.linspace(-5, 5, 100)
y = np.linspace(-4, 12, 100)
X, Y = np.meshgrid(x, y)
dtf = 1j*y
dts = 1j*x

# Calculate stability function R for each combination of dtf and dts
R = np.zeros(X.shape, dtype=np.complex128)
for i in range(np.shape(dtf)[0]):
    for j in range(np.shape(dts)[0]):
        R[i, j] = sdc_stability(dtf[i], dts[j], nodes, weights, Q, QDelta_imp, QDelta_exp, k)
R_mod = np.abs(R)

# Cap R_mod values at 1.5 to cut off any contour bigger than 1.5
R_mod_clipped = np.clip(R_mod, None, 1.5)

# Plot filled contours and the |R|=1 contour line
plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, R_mod_clipped, levels=10, cmap='viridis')
plt.colorbar(contour, label='|R(z1, z2)|')
cs = plt.contour(x, y, R_mod_clipped, levels=[1+10e-5], colors='red', linewidths=2)
plt.clabel(cs, fmt='|R|=1', colors='red')
plt.title(f'Stability region |R(z1, z2)| ≤ 1')
plt.xlabel('Re(dts)')
plt.ylabel('Im(dtf)')
plt.grid(True)
plt.show()
