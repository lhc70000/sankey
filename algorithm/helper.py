import numpy as np
from numpy import linalg as LA

np.set_printoptions(precision=16)

def clearResult(vector):
    index = sorted(range(len(vector)), key = lambda k: vector[k], reverse = True)
    index = np.array(index)
    return np.add(index, np.ones(index.shape))

def getEigen(A, i):
    a, b = LA.eig(A)
    indexList = sorted(range(len(a)), key = lambda k: a[k], reverse = True)
    index = indexList[i]
    vector = b[:, index]
    return vector

def getCrossing(inputMatrix):
    npMatrix = np.array(inputMatrix)
    p = npMatrix.shape[0]
    q = npMatrix.shape[1]
    crossing = 0
    for j in range(1, p):
        for k in range(j+1, p+1):
            for a in range(1, q):
                for b in range(a+1, q+1):
                    crossing += npMatrix[j-1, b-1] * npMatrix[k-1, a-1]
    return crossing

    