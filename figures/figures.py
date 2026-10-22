import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import pi
from mpmath import mp, mpf, sin as msin, sinh as msinh, tan, tanh, findroot
mp.dps = 25
plt.rcParams.update({'font.size': 9.5, 'font.family': 'serif',
                     'mathtext.fontset': 'cm', 'axes.linewidth': 0.7})

C_PER = '#b23434'   # periodic candidate
C_OPT = '#2b6aa8'   # competitor
GREY  = '#f2f2f2'

# ---------- Fig 1: mechanism of Theorem 1.1 (clamped, n = 2, m = 1) ----------
r2 = findroot(lambda r: tan(r)-tanh(r), 5*pi/4)
xs = np.linspace(-1, 1, 2001)
v2 = np.array([float((msinh(r2)*msin(r2*x)-msin(r2)*msinh(r2*x))/msinh(r2))
               for x in xs])**2

fig, ax = plt.subplots(figsize=(5.6, 2.95))
ax.plot(xs, v2, 'k-', lw=1.2)
for (a, b) in [(-1, -0.75), (-0.25, 0.25), (0.75, 1)]:
    ax.axvspan(a, b, ymin=0.0, ymax=0.085, color=C_PER, alpha=0.85)
ax.text(-0.985, 0.030, r'$E_{\rm per}$', color='white', fontsize=8, weight='bold')

y14 = float(v2[np.argmin(abs(xs-0.25))]); y34 = float(v2[np.argmin(abs(xs-0.75))])
ax.plot([0.25], [y14], 'o', ms=4.5, color='k')
ax.plot([0.75], [y34], 'o', ms=4.5, color='k')
ax.annotate(r'$v_2(1/4)^2$', xy=(0.25, y14), xytext=(-0.30, 0.72),
            fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.6))
ax.annotate(r'$v_2(3/4)^2$', xy=(0.75, y34), xytext=(0.40, 0.42),
            fontsize=9, arrowprops=dict(arrowstyle='-', lw=0.6))
# sliver transfer arrows (both ends)
for sgn in (+1, -1):
    ax.annotate('', xy=(sgn*0.75, 0.043), xytext=(sgn*0.25, 0.043),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color='#1a1a1a',
                                connectionstyle='arc3,rad=%.2f' % (-0.35*sgn)))
ax.text(0.0, 0.115, r'sliver transfer', fontsize=8.5, ha='center',
        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none'))
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$v_2^2/\sinh^2 r_2$')
ax.set_xlim(-1, 1); ax.set_ylim(0, 1.02)
ax.set_title(r'Clamped rod, $n=2$: the wall layer misplaces $E_{\rm per}$',
             fontsize=9.5)
fig.tight_layout(); fig.savefig('fig_clamped_sublevel.pdf'); plt.close(fig)

# ---------- Fig 2: the six certified densities at density ratio 20 ----------
fig, axs = plt.subplots(6, 1, figsize=(5.6, 3.5), sharex=True)

def drawrho(ax, blocks, color, label, lam):
    ax.axhspan(0, 1, color=GREY)
    for (a, b) in blocks:
        ax.axvspan(a, b, color=color, alpha=0.9)
    ax.set_yticks([]); ax.set_ylim(0, 1)
    ax.text(-0.985, 0.28, label, fontsize=8.5)
    ax.text(1.02, 0.28, lam, fontsize=8.5, ha='left')

per2 = [(-1, -0.75), (-0.25, 0.25), (0.75, 1)]
drawrho(axs[0], per2, C_PER, r'$\rho_{\rm per,2}$ (hinged)', r'$\lambda_2<22$')
drawrho(axs[1], [(-1, -1/6), (5/6, 1)], C_OPT, r'$\rho_*$', r'$\lambda_2>23$')
drawrho(axs[2], [(-1, -5/6), (-1/2, -1/6), (1/6, 1/2), (5/6, 1)], C_PER,
        r'$\rho_{\rm per,3}$ (hinged)', r'$\lambda_3<109$')
drawrho(axs[3], [(-1, -6/7), (-5/14, 5/14), (6/7, 1)], C_OPT,
        r'$\rho_{3\star}$', r'$\lambda_3>120$')
drawrho(axs[4], per2, C_PER, r'$\rho_{\rm per,2}$ (clamped)', r'$\Lambda_2<53$')
drawrho(axs[5], [(-1, -11/16), (-3/16, 3/16), (11/16, 1)], C_OPT,
        r'$\rho_\sharp$', r'$\Lambda_2>66$')
axs[5].set_xlabel(r'$x$'); axs[5].set_xlim(-1, 1)
axs[0].set_title(r'Contrast $\beta=20$: heavy set (shaded) and certified bound',
                 fontsize=9.5)
fig.subplots_adjust(right=0.815, hspace=0.38, top=0.90, bottom=0.13, left=0.02)
fig.savefig('fig_densities.pdf'); plt.close(fig)
print("figures written")

