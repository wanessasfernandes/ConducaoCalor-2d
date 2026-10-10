import numpy as np

def thomas(lower, diagonal, upper, rhs):
    """Resolve a matriz tridiagonal com diagonais constantes ou vetores."""
    n = len(rhs)
    c = np.asarray(upper, dtype=float).copy()
    d = np.asarray(diagonal, dtype=float).copy()
    a = np.asarray(lower, dtype=float).copy()
    b = np.asarray(rhs, dtype=float).copy()
    
    for k in range(1, n):
        factor = a[k - 1] / d[k - 1]
        d[k] -= factor * c[k - 1]
        b[k] -= factor * b[k - 1]
    
    sol = np.empty_like(b)
    sol[-1] = b[-1] / d[-1]
    
    for k in range(n - 2, -1, -1):
        sol[k] = (b[k] - c[k] * sol[k + 1]) / d[k]
    
    return sol
