"""
certificate.py:  exact rational interval certificates for two-phase rods.

Certifies sign facts about the shooting determinant
    D(lambda) = T12*T34 - T14*T32,
T = P(seg_m)* ... *P(seg_1) (segments left to right), for the hinged rod
u'''' = lambda rho u on (-1,1), u = u'' = 0 at both ends.

All internal arithmetic is exact rational interval arithmetic (PythonFractions with directed outward rounding). 
Trigonometric/hyperbolic valuesare enclosed by Taylor polynomials with explicit remainder bounds; 
fourth roots by rational bisection. Floating point appears only in printing, for visual convinience.

All densities below are admissible for (alpha,beta,c) = (1,20,21): they take the value 20 on a set of measure 1.

  rho_per,2 = 1 + 19*1_E, E = (-1,-3/4) u (-1/4,1/4) u (3/4,1)
  rho_*     = 1 + 19*1_E, E = (-1,-1/6) u (5/6,1)
  rho_per,3 = 1 + 19*1_E, E = (-1,-5/6) u (-1/2,-1/6) u (1/6,1/2) u (5/6,1)
  rho_3*    = 1 + 19*1_E, E = (-1,-6/7) u (-5/14,5/14) u (6/7,1)
  rho_sharp = 1 + 19*1_E, E = (-1,-11/16) u (-3/16,3/16) u (11/16,1)

Certified facts (matching Sections 3 and 5 of the manuscript):

  (W) clamped wall comparison: all six rows of the interface-window table,
      including the rational window containments and the uniform gap 9/100;
      and the uniform interface-ratio bounds used in the following remark.

  (A) hinged, n = 2:  D_*(7/10)>0, D_*(4/5)<0, D_*(23)<0, D_*(24)>0.
      With lambda_2 >= pi^4/20 > 4.86 and lambda_3 >= 81 pi^4/320 > 24
      this pins lambda_2(rho_*) in (23,24).

  (B) hinged, n = 3:  three sign changes of D_per, 3 below 109
      => lambda_3(rho_per,3) < 109;  and D_3* has NO zero on [24,120],
      which with lambda_3(rho_3*) > 24 forces lambda_3(rho_3*) > 120.

  (C) clamped, n = 2: two sign changes of D^cl_per,2 below 53
      => Lambda_2(rho_per,2) < 53;  and D^cl_sharp has NO zero on
      [16/5,66], which with Lambda_2 >= lambda_2 > 4.86 > 16/5 forces
      Lambda_2(rho_sharp) > 66.

No sign-change bracket is needed for the two competitors in (B) and (C):
the exclusion sweep plus the a priori comparison bound does the work, so
no eigenvalue-counting step is required.
"""

from fractions import Fraction as F

PREC = 1 << 64          # outward rounding denominator a = \fraq{b}{2**64}
NTERM = 38              # Taylor truncation order (power z^NTERM)

def require(condition, message):
    """Raise explicitly on a failed certificate condition, also under python -O."""
    if not condition:
        raise RuntimeError(message)

def rdown(x): return F((x.numerator * PREC) // x.denominator, PREC) # Rounds to closest rational with denomirator 2**64
def rup(x):   return F(-((-x.numerator * PREC) // x.denominator), PREC)

class I: # to work with intervals
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        if hi is None: hi = lo
        self.lo, self.hi = rdown(F(lo)), rup(F(hi))
        require(self.lo <= self.hi, "invalid interval")
    @staticmethod
    def _raw(lo, hi):
        v = I.__new__(I); v.lo, v.hi = rdown(lo), rup(hi); return v
    def __add__(a, b):
        if not isinstance(b, I): b = I(b)
        return I._raw(a.lo + b.lo, a.hi + b.hi)
    def __sub__(a, b):
        if not isinstance(b, I): b = I(b)
        return I._raw(a.lo - b.hi, a.hi - b.lo)
    def __mul__(a, b):
        if not isinstance(b, I): b = I(b)
        c = (a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi)
        return I._raw(min(c), max(c))
    __radd__ = __add__; __rmul__ = __mul__
    def __rsub__(a, b): return I(b) - a
    def recip(a):
        require(a.lo > 0 or a.hi < 0, "interval reciprocal contains zero")
        return I._raw(1/a.hi, 1/a.lo)
    def __truediv__(a, b):
        if not isinstance(b, I): b = I(b)
        return a * b.recip()
    def sign(a):
        if a.lo > 0: return 1
        if a.hi < 0: return -1
        return 0
    def __repr__(self):
        return f"[{decimal_down(self.lo)},{decimal_up(self.hi)}]"

def _fixed_decimal(n, digits=8): # decimal representation of rational fraction (up to 8 digits precision)
    scale = 10 ** digits
    sign = '-' if n < 0 else ''
    whole, frac = divmod(abs(n), scale)
    return f"{sign}{whole}.{frac:0{digits}d}"

def decimal_down(x, digits=8): #Fixed-point decimal rounded toward minus infinity
    scale = 10 ** digits
    return _fixed_decimal((x.numerator * scale) // x.denominator, digits)

def decimal_up(x, digits=8): #Fixed-point decimal rounded toward plus infinity
    scale = 10 ** digits
    return _fixed_decimal(-((-x.numerator * scale) // x.denominator), digits)

def root4(v): #interval enclosure of v^(1/4) for interval v with v.lo > 0. Integer bisection: find N with (N/S)^4 <= x < ((N+1)/S)^4
    S = 1 << 40
    def r4(x, up):
        p, q = x.numerator, x.denominator
        lo, hi = 0, S * (1 + p // q + 1)
        while hi - lo > 1:
            m = (lo + hi) // 2
            if m**4 * q <= p * S**4: lo = m
            else: hi = m
        return F(hi, S) if up else F(lo, S)
    return I._raw(r4(v.lo, False), r4(v.hi, True))

def _taylor(z, kind): #kind: 'sin','cos','sinh','cosh'. Valid for |z| <= 8
    zmax = max(abs(z.lo), abs(z.hi))
    require(zmax <= 8, "argument too large for fixed Taylor budget")
    s = I(0); term = I(1); zz = z * z
    if kind in ('sin', 'sinh'):
        term = z
        p = 1
    else:
        p = 0
    j = 0
    sgn = 1
    while p <= NTERM:
        s = s + (term if (kind in ('sinh','cosh') or sgn > 0) else I(0) - term)
        term = term * zz
        term = term * F(1, (p+1)*(p+2))
        p += 2
        if kind in ('sin','cos'): sgn = -sgn
    # remainder is: sum_{k>=p} |z|^k/k! <= 2*zmax^p/p!  (since zmax <= 8 <= (p+1)/2)
    num = F(1)
    zm = rup(F(zmax))
    for k in range(1, p+1): num = rup(num * zm / k)
    R = 2 * num
    return I._raw(s.lo - R, s.hi + R)

def sinI(z):  return _taylor(z, 'sin')
def cosI(z):  return _taylor(z, 'cos')
def sinhI(z): return _taylor(z, 'sinh')
def coshI(z): return _taylor(z, 'cosh')

def exp_minus_I(x):
    z = I(x)
    return coshI(z) - sinhI(z)

def H1_I(x): #H_1(x)=exp(-x)+sin(x)-cos(x), evaluated at a rational point
    z = I(x)
    return coshI(z) - sinhI(z) + sinI(z) - cosI(z)

def eta_I(x): #The error majorant 10^{-3}(sinh(x)+1)
    return (sinhI(I(x)) + 1) * F(1, 1000)

def certify_wall_table():

    #Certify the scalar profile estimates, Table 1, and the ratio remark"
    # Scalar Taylor bounds used in the proof of the shape lemma.

    require(sinI(I(F(2386, 1000))).lo > F(685, 1000), "sin(2.386) bound")
    require(sinI(I(F(2786, 1000))).lo > F(348, 1000), "sin(2.786) bound")
    require(sinI(I(F(3036, 1000))).lo > F(105, 1000), "sin(3.036) bound")
    require(cosI(I(F(9, 4))).hi < -F(628, 1000), "cos(2.25) bound")
    require(sinI(I(F(19, 8))).lo > F(693, 1000), "sin(2.375) bound")
    require(exp_minus_I(F(9, 4)).hi < F(106, 1000), "exp(-2.25) bound")

    # Classical bounds 333/106 < pi < 355/113 turn the symbolic theta
    # interval in (3.8) into rational outer bounds.

    theta_lo = F(333, 106) - F(1, 500)
    theta_hi = F(5, 4) * F(355, 113) + F(1, 500)
    rows = [
        (F(1,2), F(5,8), F(392,1000), F(737,1000),
         F(2648,1000), F(3561,1000), F(1697,10000), F(2661,10000), F(964,10000)),
        (F(5,8), F(3,4), F(490,1000), F(860,1000),
         F(2550,1000), F(3438,1000), F(2815,10000), F(4622,10000), F(1807,10000)),
        (F(3,4), F(7,8), F(588,1000), F(983,1000),
         F(2452,1000), F(3316,1000), F(4277,10000), F(6936,10000), F(2659,10000)),
        (F(7,8), F(1,1), F(686,1000), F(1106,1000),
         F(2354,1000), F(3193,1000), F(6067,10000), F(9509,10000), F(3442,10000)),
        (F(1,1), F(5,4), F(784,1000), F(1351,1000),
         F(2158,1000), F(3070,1000), F(10398,10000), F(12180,10000), F(1782,10000)),
        (F(5,4), F(3,2), F(981,1000), F(1597,1000),
         F(1962,1000), F(2824,1000), F(15172,10000), F(17219,10000), F(2047,10000)),
    ]
    for index, row in enumerate(rows, 1):
        m_lo, m_hi, p_lo, p_hi, q_lo, q_hi, a_cap, b_floor, gap_floor = row
        actual_p_lo = theta_lo * m_lo / 4
        actual_p_hi = theta_hi * (m_hi / 4 + F(1, 32))
        actual_q_lo = theta_lo * (1 - m_hi / 4)
        actual_q_hi = theta_hi * (1 - m_lo / 4 + F(1, 32))
        require(p_lo <= actual_p_lo and actual_p_hi <= p_hi,
                f"wall table row {index}: first window containment")
        require(q_lo <= actual_q_lo and actual_q_hi <= q_hi,
                f"wall table row {index}: second window containment")

        a = H1_I(p_hi) + eta_I(p_hi)
        require(a.lo > 0, f"wall table row {index}: a positivity")
        a2_hi = (a * a).hi
        hq_lo, hq_hi = H1_I(q_lo), H1_I(q_hi)
        b_lo = min(hq_lo.lo, hq_hi.lo) - eta_I(q_hi).hi
        require(b_lo > 0, f"wall table row {index}: b positivity")
        b2_lo = b_lo * b_lo
        require(a2_hi <= a_cap, f"wall table row {index}: a^2 bound")
        require(b2_lo >= b_floor, f"wall table row {index}: b^2 bound")
        require(b_floor - a_cap >= gap_floor >= F(9, 100),
                f"wall table row {index}: gap bound")
        print(f"  row {index}: a^2 <= {decimal_up(a2_hi)}, "
              f"b^2 >= {decimal_down(b2_lo)}, gap >= {gap_floor}")

    # Uniform interface-ratio remark, using its displayed outer windows.

    p_lo, p_hi = F(784,1000), F(983,1000)
    q_lo, q_hi = F(2354,1000), F(2947,1000)
    require(p_lo <= theta_lo / 4 and theta_hi / 4 <= p_hi,
            "ratio: first window containment")
    require(q_lo <= 3 * theta_lo / 4 and 3 * theta_hi / 4 <= q_hi,
            "ratio: second window containment")
    a = H1_I(p_hi) + eta_I(p_hi)
    b_lo = min(H1_I(q_lo).lo, H1_I(q_hi).lo) - eta_I(q_hi).hi
    a2_hi, b2_lo = (a * a).hi, b_lo * b_lo
    require(a2_hi <= F(428,1000), "ratio: numerator bound")
    require(b2_lo >= F(1479,1000), "ratio: denominator bound")
    require(F(428,1000) / F(1479,1000) < F(3,10), "ratio: quotient bound")
    print(f"  ratio: numerator <= {decimal_up(a2_hi)}, "
          f"denominator >= {decimal_down(b2_lo)}, quotient < 3/10")

def Pmat(rho, ell, lam): #transfer matrix over a segment of density rho (int), length ell (Fraction),lam an interval
    k = root4(lam * rho)
    z = k * F(ell)
    c, s, C, S = cosI(z), sinI(z), coshI(z), sinhI(z)
    k2 = k * k; k3 = k2 * k
    half = F(1, 2)
    return [
        [ (c+C)*half, (s+S)*half/k, (C-c)*half/k2, (S-s)*half/k3 ],
        [ k*(S-s)*half, (c+C)*half, (s+S)*half/k, (C-c)*half/k2 ],
        [ k2*(C-c)*half, k*(S-s)*half, (c+C)*half, (s+S)*half/k ],
        [ k3*(S+s)*half, k2*(C-c)*half, k*(S-s)*half, (c+C)*half ] ]

def matmul(A, B): # regular mtx multiplication for 4 by 4 matrices
    return [[sum((A[i][t]*B[t][j] for t in range(4)), I(0)) for j in range(4)]
            for i in range(4)]

def Dchar(segs, lam, bc='hinged'):
    """segs: list of (rho, ell) LEFT to RIGHT; lam interval.
    hinged : initial (0,a,0,b), final u = u'' = 0 -> rows 1,3 / cols 2,4
    clamped: initial (0,0,a,b), final u = u'  = 0 -> rows 1,2 / cols 3,4"""
    T = None
    for rho, ell in segs:
        P = Pmat(rho, ell, lam)
        T = P if T is None else matmul(P, T)
    if bc == 'hinged':
        return T[0][1]*T[2][3] - T[0][3]*T[2][1]
    return T[0][2]*T[1][3] - T[0][3]*T[1][2]

def certify_sign(segs, lam_lo, lam_hi, want, label, bc='hinged'):
    d = Dchar(segs, I(F(lam_lo), F(lam_hi)), bc)
    ok = d.sign() == want
    print(f"  {label}: D([{lam_lo},{lam_hi}]) in {d}  sign {'OK' if ok else 'FAIL'}")
    require(ok, label)
    return d

class DI:
    """dual interval: value and d/dlambda, both intervals (forward mode)."""
    __slots__ = ('v', 'd')
    def __init__(self, v, d):
        self.v, self.d = v, d
    def __add__(a, b):
        if not isinstance(b, DI): b = DI(I(b), I(0))
        return DI(a.v + b.v, a.d + b.d)
    __radd__ = __add__
    def __sub__(a, b):
        if not isinstance(b, DI): b = DI(I(b), I(0))
        return DI(a.v - b.v, a.d - b.d)
    def __mul__(a, b):
        if not isinstance(b, DI):
            bb = I(b)
            return DI(a.v * bb, a.d * bb)
        return DI(a.v * b.v, a.d * b.v + a.v * b.d)
    __rmul__ = __mul__
    def __truediv__(a, b):
        if not isinstance(b, DI):
            bb = I(b).recip()
            return DI(a.v * bb, a.d * bb)
        r = b.v.recip()
        return DI(a.v * r, (a.d * b.v - a.v * b.d) * (r * r))

def Pmat_dual(rho, ell, lam):
    """transfer matrix with d/dlambda, lam an interval (possibly a box)."""
    k_v = root4(lam * rho)
    k = DI(k_v, k_v * lam.recip() * F(1, 4))       # dk/dlam = k/(4 lam)
    z = k * F(ell)
    c_v, s_v = cosI(z.v), sinI(z.v)
    C_v, S_v = coshI(z.v), sinhI(z.v)
    c = DI(c_v, I(0) - s_v * z.d)
    s = DI(s_v, c_v * z.d)
    C = DI(C_v, S_v * z.d)
    S = DI(S_v, C_v * z.d)
    half = F(1, 2)
    return [
        [ (c+C)*half, (s+S)*half/k, (C-c)*half/(k*k), (S-s)*half/(k*k*k) ],
        [ k*(S-s)*half, (c+C)*half, (s+S)*half/k, (C-c)*half/(k*k) ],
        [ (k*k)*(C-c)*half, k*(S-s)*half, (c+C)*half, (s+S)*half/k ],
        [ (k*k*k)*(S+s)*half, (k*k)*(C-c)*half, k*(S-s)*half, (c+C)*half ] ]

def Dchar_dual(segs, lam, bc='hinged'):
    T = None
    for rho, ell in segs:
        P = Pmat_dual(rho, ell, lam)
        if T is None:
            T = P
        else:
            T = [[sum((P[i][t]*T[t][j] for t in range(4)), DI(I(0), I(0)))
                  for j in range(4)] for i in range(4)]
    if bc == 'hinged':
        return T[0][1]*T[2][3] - T[0][3]*T[2][1]
    return T[0][2]*T[1][3] - T[0][3]*T[1][2]

def certify_no_root(segs, lo, hi, label, depth=0, maxdepth=48, bc='hinged'):
    """Prove D != 0 on [lo,hi] by adaptive bisection with the mean-value
    (slope) form:  D(lam) in D(m) + (lam-m)*D'([lo,hi]) for m=(lo+hi)/2,
    the derivative enclosed by forward-mode differentiation in the same
    interval arithmetic.  Falls back to direct interval evaluation on
    narrow boxes."""
    lo, hi = F(lo), F(hi)
    m = (lo + hi) / 2
    w2 = (hi - lo) / 2
    dm = Dchar(segs, I(m, m), bc)                   # point value at midpoint
    dp = Dchar_dual(segs, I(lo, hi), bc).d          # derivative over the box
    rad = max(abs(dp.lo), abs(dp.hi)) * w2
    encl = I._raw(dm.lo - rad, dm.hi + rad)
    if encl.sign() != 0:
        return 1
    if depth >= maxdepth:
        raise RuntimeError(f"exclusion failed near [{float(lo)},{float(hi)}]")
    n1 = certify_no_root(segs, lo, m, label, depth+1, maxdepth, bc)
    n2 = certify_no_root(segs, m, hi, label, depth+1, maxdepth, bc)
    return n1 + n2

# ---------------- densities ----------------
# n = 2
per2  = [(20, F(1,4)), (1, F(1,2)), (20, F(1,2)), (1, F(1,2)), (20, F(1,4))]
star2 = [(20, F(5,6)), (1, F(1,1)), (20, F(1,6))]
# n = 3
per3  = [(20, F(1,6)), (1, F(1,3)), (20, F(1,3)), (1, F(1,3)),
         (20, F(1,3)), (1, F(1,3)), (20, F(1,6))]
star3 = [(20, F(1,7)), (1, F(1,2)), (20, F(5,7)), (1, F(1,2)), (20, F(1,7))]
# clamped, n = 2:  E_per as above;  E_sharp = (-1,-11/16) u (-3/16,3/16) u (11/16,1)
sharp2 = [(20, F(5,16)), (1, F(1,2)), (20, F(3,8)), (1, F(1,2)), (20, F(5,16))]

def main():
    print("== (W) clamped wall profile and six-row interface table ==")
    certify_wall_table()

    print("== (A) n = 2 : asymmetric competitor rho_* ==")
    certify_sign(star2, F(7,10), F(7,10), +1, "D(7/10) > 0")
    certify_sign(star2, F(4,5),  F(4,5),  -1, "D(4/5)  < 0")
    certify_sign(star2, 23, 23, -1, "D(23)   < 0")
    certify_sign(star2, 24, 24, +1, "D(24)   > 0")

    print("== (B) n = 3 : periodic density, three eigenvalue brackets below 109 ==")
    certify_sign(per3, F(11,20), F(11,20), +1, "D(0.55) > 0")
    certify_sign(per3, F(3,5),   F(3,5),   -1, "D(0.60) < 0")
    certify_sign(per3, 9, 9, -1, "D(9)   < 0")   # signs fixed after first run
    certify_sign(per3, F(91,10), F(91,10), +1, "D(9.1) > 0")
    certify_sign(per3, 108, 108, +1, "D(108) > 0")
    certify_sign(per3, 109, 109, -1, "D(109) < 0")

    print("== (B) n = 3 : competitor rho_3*, exclusion sweep on [24,120] ==")
    n = certify_no_root(star3, F(24), F(120), "[24,120]")
    print(f"  no root of D_3* in [24,120]  ({n} boxes)")

    print("== (C) clamped n = 2 : periodic density, two brackets below 53 ==")
    certify_sign(per2, F(5,2), F(5,2), +1, "D(2.5)  > 0", bc='clamped')
    certify_sign(per2, F(13,5), F(13,5), -1, "D(2.6)  < 0", bc='clamped')
    certify_sign(per2, 52, 52, -1, "D(52)   < 0", bc='clamped')
    certify_sign(per2, 53, 53, +1, "D(53)   > 0", bc='clamped')

    print("== (C) clamped n = 2 : competitor rho_sharp, exclusion sweep on [16/5,66] ==")
    m = certify_no_root(sharp2, F(16,5), F(66), "[3.2,66]", bc='clamped')
    print(f"  no root of D_sharp in [3.2,66]  ({m} boxes)")

    print(f"  exclusion totals: {n} boxes (hinged n=3), {m} boxes (clamped n=2)")
    print("ALL CERTIFICATES PASSED")

if __name__ == "__main__":
    main()
