import numpy as np
from numpy import linalg as LA
import operator
import helper

np.set_printoptions(suppress=True)

def randomMatrix(matrix, beta):
    orig = np.dot((1 - beta), matrix)
    rand = np.random.rand(matrix.shape[0], matrix.shape[1])
    for i, arr in enumerate(rand):
        sum = np.sum(rand[i])
        rand[i] = np.dot(beta/sum, arr)
    return np.add(orig, rand)

def parallel(matrixList, beta=0.1, eigen = 1):
    newMatrixList = []
    for i,matrix in enumerate(matrixList):
        matrix = np.array(matrix)
        newMatrixList.append(randomMatrix(matrix, beta))
        if i == 0:
            A = newMatrixList[i]
        else:
            A = np.matmul(A, newMatrixList[i])
    vector = helper.getEigen(A, eigen)
    vectors = []
    result = []
    for i,matrix in enumerate(newMatrixList):
        if i < len(newMatrixList)/2 + 1:
            if i == 0:
                vectors.append(vector)
                result.append(helper.clearResult(vector))
                vector = np.dot(newMatrixList[len(newMatrixList) - i - 1], vector)
            else:
                vectors.append(vector)
                result.append(helper.clearResult(vector))
                vector = np.dot(newMatrixList[len(newMatrixList) - i - 1], vector)
    return {
        "result": result,
        "vectors": vectors,
    }


