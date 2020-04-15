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

def Fock(Da, Db, Hcore, ERI, dimension):
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

def Unitary(mat):
    return np.allclose(np.linalg.inv(mat),np.transpose(mat))

def Hermitian(mat):
    return np.allclose(np.transpose(mat), mat)

def EHF(Hcore, Da, Fa, Db, Fb):
    alfa = Hcore + Fa
    beta = Hcore + Fb
    suma = np.einsum('ij,ij->', Da, alfa)
    sumb = np.einsum('ij,ij->', Db, beta)
    return (suma + sumb)/2