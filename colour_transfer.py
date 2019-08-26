import numpy as np
import scipy as sp

import logging

log = logging.getLogger('colour_transfer')

eps = np.finfo(float).eps

def colour_transfer_mkl(x0, x1):
    a = np.cov(x0.T)
    b = np.cov(x1.T)

    Da2, Ua = np.linalg.eig(a)
    Da = np.diag(np.sqrt(Da2.clip(eps, None))) 

    C = np.dot(np.dot(np.dot(np.dot(Da, Ua.T), b), Ua), Da)

    Dc2, Uc = np.linalg.eig(C)
    Dc = np.diag(np.sqrt(Dc2.clip(eps, None))) 

    Da_inv = np.diag(1./(np.diag(Da)))

    t = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(Ua, Da_inv), Uc), Dc), Uc.T), Da_inv), Ua.T) 

    mx0 = np.mean(x0, axis=0)
    mx1 = np.mean(x1, axis=0)

    return np.dot(x0-mx0, t) + mx1
    
def colour_transfer_idt(i0, i1, bins=300, n_rot=10, relaxation=1):
    n_dims = i0.shape[1]
    
    d0 = i0.T
    d1 = i1.T
    
    for i in range(n_rot):
        
        r = sp.stats.special_ortho_group.rvs(n_dims).astype(np.float32)
        
        d0r = np.dot(r, d0)
        d1r = np.dot(r, d1)
        d_r = np.empty_like(d0)
        
        for j in range(n_dims):
            
            lo = min(d0r[j].min(), d1r[j].min())
            hi = max(d0r[j].max(), d1r[j].max())
            
            p0r, edges = np.histogram(d0r[j], bins=bins, range=[lo, hi])
            p1r, _     = np.histogram(d1r[j], bins=bins, range=[lo, hi])

            cp0r = p0r.cumsum().astype(np.float32)
            cp0r /= cp0r[-1]

            cp1r = p1r.cumsum().astype(np.float32)
            cp1r /= cp1r[-1]
            
            f = np.interp(cp0r, cp1r, edges[1:])
            
            d_r[j] = np.interp(d0r[j], edges[1:], f, left=0, right=bins)
        
        d0 = relaxation * np.linalg.solve(r, (d_r - d0r)) + d0
        
        log.debug('Iteration %d/%d completed.', i+1, n_rot)
    
    return d0.T