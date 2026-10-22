import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

def elem_mats(h):
    K = (1.0/h**3)*np.array([[12,6*h,-12,6*h],[6*h,4*h*h,-6*h,2*h*h],
                             [-12,-6*h,12,-6*h],[6*h,2*h*h,-6*h,4*h*h]])
    M = (h/420.0)*np.array([[156,22*h,54,-13*h],[22*h,4*h*h,13*h,-3*h*h],
                            [54,13*h,156,-22*h],[-13*h,-3*h*h,-22*h,4*h*h]])
    return K, M

_cache = {}
def modes_uniform(rho_elem, bc, k):
    """rho_elem: array of length N of element densities on uniform mesh over (-1,1)."""
    N = len(rho_elem)
    key = N
    if key not in _cache:
        h = 2.0/N
        Ke, Me = elem_mats(h)
        rows = np.empty(16*N, dtype=np.int32); cols = np.empty(16*N, dtype=np.int32)
        kv = np.empty(16*N); mv = np.empty(16*N)
        idx = np.arange(4)
        for e in range(N):
            base = 2*e
            I, J = np.meshgrid(base+idx, base+idx, indexing='ij')
            rows[16*e:16*e+16] = I.ravel(); cols[16*e:16*e+16] = J.ravel()
            kv[16*e:16*e+16] = Ke.ravel(); mv[16*e:16*e+16] = Me.ravel()
        _cache[key] = (rows, cols, kv, mv)
    rows, cols, kv, mv = _cache[key]
    mvr = mv * np.repeat(rho_elem, 16)
    ndof = 2*(N+1)
    K = sparse.coo_matrix((kv,(rows,cols)), shape=(ndof,ndof)).tocsr()
    M = sparse.coo_matrix((mvr,(rows,cols)), shape=(ndof,ndof)).tocsr()
    fixed = [0, 2*N] if bc=='hinged' else [0,1,2*N,2*N+1]
    keep = np.setdiff1d(np.arange(ndof), fixed)
    K = K[keep][:,keep]; M = M[keep][:,keep]
    vals, vecs = eigsh(K, k=k, M=M, sigma=0, which='LM')
    o = np.argsort(vals)
    U = np.zeros((ndof, k)); U[keep,:] = vecs[:,o]
    return vals[o], U[0::2,:]

def rho_from_blocks(blocks, beta, N):
    xm = -1 + (np.arange(N)+0.5)*(2.0/N)
    r = np.ones(N)
    for a,b in blocks:
        r[(xm>=a)&(xm<=b)] = beta
    return r

def blocks_from_ind(x, ind):
    blocks=[]; a=None
    for i,f in enumerate(ind):
        if f and a is None: a=x[i]
        if (not f) and a is not None: blocks.append((a,x[i])); a=None
    if a is not None: blocks.append((a,x[-1]))
    return blocks

def bathtub(mode, bc, beta, blocks0, N=400, iters=60, tol=1e-11):
    blocks = list(blocks0); lamprev = None
    x = np.linspace(-1,1,N+1); xm = 0.5*(x[:-1]+x[1:]); w = 2.0/N
    for it in range(iters):
        rho = rho_from_blocks(blocks, beta, N)
        vals, defl = modes_uniform(rho, bc, mode+1)
        u2 = 0.5*(defl[:-1,mode-1]**2 + defl[1:,mode-1]**2)
        order = np.argsort(u2)
        nheavy = int(round(1.0/w))
        sel = np.zeros(N, bool); sel[order[:nheavy]] = True
        blocks = blocks_from_ind(x, np.append(sel, sel[-1]))
        if lamprev is not None and abs(vals[mode-1]-lamprev) < tol:
            lamprev = vals[mode-1]; break
        lamprev = vals[mode-1]
    rho = rho_from_blocks(blocks, beta, N)
    vals, defl = modes_uniform(rho, bc, mode+2)
    return vals, blocks

# exact transfer matrix (float) for final evaluations
def Pmat(rho, ell, lam):
    kk = (lam*rho)**0.25; z = kk*ell
    c,s,C,S = np.cos(z), np.sin(z), np.cosh(z), np.sinh(z)
    return np.array([[(c+C)/2,(s+S)/(2*kk),(C-c)/(2*kk*kk),(S-s)/(2*kk**3)],
                     [kk*(S-s)/2,(c+C)/2,(s+S)/(2*kk),(C-c)/(2*kk*kk)],
                     [kk*kk*(C-c)/2,kk*(S-s)/2,(c+C)/2,(s+S)/(2*kk)],
                     [kk**3*(S+s)/2,kk*kk*(C-c)/2,kk*(S-s)/2,(c+C)/2]])
def Dchar(segs, lam, bc='hinged'):
    T = np.eye(4)
    for rho,ell in segs:
        if ell>1e-13: T = Pmat(rho,ell,lam)@T
    return T[0,1]*T[2,3]-T[0,3]*T[2,1] if bc=='hinged' else T[0,2]*T[1,3]-T[0,3]*T[1,2]
def segs_from_blocks(blocks, beta):
    segs=[]; xx=-1.0
    for a,b in sorted(blocks):
        if a>xx: segs.append((1.0,a-xx))
        segs.append((beta,b-a)); xx=b
    if xx<1.0: segs.append((1.0,1.0-xx))
    return segs
def eig_tm(blocks, beta, k, bc='hinged', lam_max=500.0, step=0.02):
    segs = segs_from_blocks(blocks, beta)
    out=[]; lam=1e-8; prev=Dchar(segs,lam,bc)
    while lam<lam_max and len(out)<k:
        lam2=lam+step; cur=Dchar(segs,lam2,bc)
        if np.sign(cur)!=np.sign(prev):
            a,b=lam,lam2
            for _ in range(60):
                m=0.5*(a+b)
                if np.sign(Dchar(segs,m,bc))==np.sign(Dchar(segs,a,bc)): a=m
                else: b=m
            out.append(0.5*(a+b))
        lam,prev=lam2,cur
    while len(out)<k: out.append(np.nan)
    return out

