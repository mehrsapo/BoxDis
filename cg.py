from __init__ import * 


def cg(A, b, x, max_iter=1000, tol=1e-10):
    r = b - A(x)
    p = r
    for i in range(max_iter):
        alpha = r.norm()**2 / (p * A(p)).sum()
        x = x + alpha * p
        r_old = torch.clone(r)
        r = r - alpha * A(p)
        if (r).norm() < tol:
            break
        beta = r.norm()**2 / r_old.norm()**2
        p = r + beta * p

    return x 