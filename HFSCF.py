import numpy as np
print('Imported HFSCF')

def OrthoS(eigenval, eigenvec):
    return eigenvec @ (np.linalg.inv(eigenval) ** 0.5) @ np.transpose(eigenvec)

def Diag(mat):
    eigenval = np.diag(np.linalg.eigh(mat)[0])
    eigenvec = np.linalg.eigh(mat)[1]
    return eigenval, eigenvec

def OrthoF(eigenval, eigenvec):
    return np.transpose(eigenvec) @ eigenval @ eigenvec

def fock(D, Hcore, ERI, dimension):
    F = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            sum = 0
            F[i][j] += Hcore[i][j]
            for k in range(dimension):
                for l in range(dimension):
                    sum += D[k][l] * ((2 * ERI[i][j][k][l]) - ERI[i][k][j][l])
            F[i][j] += sum
    return F

def ElEnergy(D, Hcore, Fock):
    sum = Hcore + Fock
    return np.einsum('ij,ij->', D, sum)
