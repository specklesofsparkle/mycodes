import numpy as np

# A method performing least squares approximation of a linear courve.
# Input: Vectors x,y. Both 1D np.array of same size.
# Output: list of factors [m, b] representing the linear courve f(x) = mx + b.
def linearLSQ(x: np.array, y: np.array) -> list:
    x_sq = np.square(x)
    xy = np.multiply(x,y)
    m = (np.multiply(xy.size,xy.sum()) - np.multiply(x.sum(),y.sum()))/(x.size*x_sq.sum() - np.square(x.sum()))
    b = (y.sum() - m*x.sum())/x.size
    return [m, b]

# A method, orthogornalizing the given basis.
# Input: sourceBase - list of vectors, as in a)
# Output: orthoronalizedBase - list of vectors, with same size and shape as sourceBase
def orthonormalize(sourceBase: list) -> list:
    x = sourceBase
    v = []
    vbase = []
    for i in range(len(x)):
        vtemp = x[i]
        for n in range(i):
            proj = np.multiply(
                np.divide(
                    np.inner(x[i], v[n]),
                    np.inner(v[n], v[n])
                ),
                v[n]
            )
            vtemp = vtemp - proj
        v.append(vtemp)
        vbase.append(np.divide(vtemp, np.linalg.norm(vtemp)))
    return vbase

