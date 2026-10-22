#!/usr/bin/env python3
"""
reproduce.py -- regenerates the non-certified numerical values quoted in
the paper and independently recomputes Table 1.  The rigorous Table 1
verification is performed by certificate.py.

Requires numpy and scipy.  Usage:  python3 reproduce.py
"""
import numpy as np
from math import pi, sin, sinh, tan, tanh
from scipy.optimize import brentq
from fast import eig_tm

per2   = [(-1, -0.75), (-0.25, 0.25), (0.75, 1)]
star2  = [(-1, -1/6), (5/6, 1)]
per3   = [(-1, -5/6), (-1/2, -1/6), (1/6, 1/2), (5/6, 1)]
star3  = [(-1, -6/7), (-5/14, 5/14), (6/7, 1)]
sharp2 = [(-1, -11/16), (-3/16, 3/16), (11/16, 1)]

print("== Section 5, closing remark: refined eigenvalues at density ratio 20 ==")
print("  lambda_2(rho_per,2)  =", eig_tm(per2,  20, 2)[1])
print("  lambda_2(rho_*)      =", eig_tm(star2, 20, 2)[1])
print("  lambda_3(rho_per,3)  =", eig_tm(per3,  20, 3, lam_max=300)[2])
print("  lambda_3(rho_3star)  =", eig_tm(star3, 20, 3, lam_max=300)[2])
print("  Lambda_2(rho_per,2)  =", eig_tm(per2,   20, 2, 'clamped', lam_max=200)[1])
print("  Lambda_2(rho_sharp)  =", eig_tm(sharp2, 20, 2, 'clamped', lam_max=200)[1])
print("  (certified: <22, >23, <109, >120, <53, >66)")

print()
print("== Section 3, \"by hand\" remark: clamped n = 2 interface values ==")
r2 = brentq(lambda r: tan(r) - tanh(r), pi + 1e-8, 3*pi/2 - 1e-8)
v = lambda x: sinh(r2)*sin(r2*x) - sin(r2)*sinh(r2*x)
print("  r_2        =", r2)
print("  r_2^4      =", r2**4)
print("  v(1/4)^2   =", v(1/4)**2)
print("  v(3/4)^2   =", v(3/4)**2)
print("  ratio      =", (v(3/4)/v(1/4))**2, " (uniform-ratio remark: <= 3/10)")

print()
print("== Sharper floating-point values underlying Table 1 ==")
H1  = lambda p: np.exp(-p) + np.sqrt(2)*np.sin(p - np.pi/4)
eta = lambda p: 1e-3*(np.sinh(p) + 1)
thm, thp = np.pi - 2e-3, 5*np.pi/4 + 2e-3
print("   m-range        phi-window        phi'-window       a^2      b^2    b^2-a^2")
for a_, b_ in [(0.5,0.625), (0.625,0.75), (0.75,0.875), (0.875,1.0),
               (1.0,1.25), (1.25,1.5)]:
    p_lo, p_hi = thm*a_/4, thp*(b_/4 + 1/32)
    q_lo, q_hi = thm*(1 - b_/4), thp*(1 - a_/4 + 1/32)
    A = (H1(p_hi) + eta(p_hi))**2
    B = (min(H1(q_lo), H1(q_hi)) - eta(q_hi))**2
    if not (p_hi <= 8/5 and q_hi <= 18/5 and B - A >= 9/100):
        raise RuntimeError("floating-point Table 1 check failed")
    print("  [%5.3f,%5.3f] [%.3f,%.3f] [%.3f,%.3f] %8.4f %8.4f %8.4f"
          % (a_, b_, p_lo, p_hi, q_lo, q_hi, A, B, B - A))
print("  all six rows satisfy b^2 - a^2 >= 9/100")

print()
print("== window containment theta = r_n h in [pi-2e-3, 5pi/4+2e-3] ==")
worst = (10, 0)
for n in list(range(2, 60)) + [100, 200, 400]:
    g = (2*n + 1)*pi/4
    f = (lambda r: tan(r) + tanh(r)) if n % 2 else (lambda r: tan(r) - tanh(r))
    th = brentq(f, g - 0.01, g + 0.01) * 2/n
    if not (thm <= th <= thp):
        raise RuntimeError((n, th))
    worst = (min(worst[0], th), max(worst[1], th))
print("  theta ranges over [%.6f, %.6f]; bounds are [%.6f, %.6f]  OK"
      % (worst[0], worst[1], thm, thp))
print()
print("done.")
