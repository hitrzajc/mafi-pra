import numpy as np
from numpy import linalg
import scipy
import math

import scipy.linalg
pow = math.pow

class Matrix:        
    def __init__(self, N, λ):
        self.matrix = np.zeros((N,N))
        self.λ = λ #parameter of
        self.N = N #matrix size
        self.fillings = [self.fill1, self.fill2, self.fill3]
        self.algorithms = [linalg.eigvals, linalg.eig, scipy.linalg.eigvals,
                           self.QR_trid, self.QR]

    def empty(self):
        N = self.N
        self.matrix = np.zeros((N,N))

    def fill(self): #dodamo navaden hamiltinjan
        for i in range(self.N):
            self.matrix[i][i] += i + 1/2

    def fill1(self):
        N = self.N
        for i in range(N):
            j = i-1
            if j>=0:
                self.matrix[i][j] = pow(1/2 * np.sqrt(i+j+1),4)
                # self.matrix[j][i] = self.matrix[i][j]
            j = i+1
            if j<N:
                self.matrix[i][j] = pow(1/2 * np.sqrt(i+j+1),4)
        # self.matrix = self.matrix @ self.matrix
        # self.matrix = self.matrix @ self.matrix
        self.matrix = self.λ * self.matrix
        self.fill()

    def fill2(self):
        N = self.N
        for i in range(N):
            self.matrix[i][i] = pow(1/2 * (2*i+1),2)
            j = i+2
            if j < N:
                self.matrix[i][j] = pow(1/2 * np.sqrt(j*(j-1)),2)
            j = i-2
            if j > 0:
                self.matrix[i][j] = pow(1/2 * np.sqrt((j+1)*(j+2)),2)
        # self.matrix =self.matrix @ self.matrix
        self.matrix = self.λ *self.matrix
        self.fill()

    def fill3(self):
        N = self.N
        pi = 1
        faci = 1
        for i in range(N):
            if i != 0:
                faci *= i
                pi *= 2
            pj = 1
            facj = 1
            #for j in range(max(0,i-4), min(i+5,N)):
            for j in range(N):
                if j != 0:
                    facj *= j
                    pj *= 2
                if abs(j-i) > 5:
                    continue
                factor = np.sqrt(pi * faci / (pj * facj))/16
                if i == j+4:
                   self.matrix[i][j]=factor
                elif i==j+2:
                   self.matrix[i][j]=factor*4*(2*j+3)
                elif i==j:
                   self.matrix[i][j]=factor*12*(2*j*j+2*j+1)
                elif i==j-2:
                   self.matrix[i][j]=factor*16*j*(2*j*j-3*j+1)
                elif i==j-4:
                   self.matrix[i][j]=factor*16*j*(j**3 - 6*j*j + 11*j - 6)
        self.matrix = self.λ *self.matrix
        self.fill()
    
    def eigvals(self):
        return linalg.eigvals(self.matrix)
    
    def eigh(self):
        return linalg.eigh(self.matrix)
    def eig(self):
        return linalg.eig(self.matrix)

    def qr(self, M):
        A = np.copy(M)
        m, n = A.shape
        Q = np.eye(m)
        for i in range(n - (m == n)):
            H = np.eye(m)
            H[i:, i:] = self.make_householder(A[i:, i])
            Q = np.dot(Q, H)
            A = np.dot(H, A)
        return Q, A

    def make_householder(self, a):
    #rescaling to v and sign for numerical stability
        v = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
        v[0] = 1
        H = np.eye(a.shape[0])
        H -= (2 / np.dot(v, v)) * np.outer(v,v)
        return H

    def qr_givens(self, M):
        A = np.copy(M)
        m, n = A.shape
        Q = np.eye(m)
        for j in range(n - (m == n)):
            for i in range(j+1,m):
                r=np.hypot(A[j,j],A[i,j])
                c=A[j,j]/r
                s=A[i,j]/r
                givensRot = np.array([[c, s],[-s,  c]])
                A[[j,i],j:] = np.dot(givensRot, A[[j,i],j:])
                Q[[j,i],:] = np.dot(givensRot, Q[[j,i],:])
        return Q.T, A

    def trid_householder(self,M):
        A = np.copy(M)
        m, n = A.shape
        if ( m != n):
            print("need quadratic symmetric matrix")
            # sys.exit(1)
            return 0
        Q = np.eye(m)
        for i in range(m - 2):
            H = np.eye(m)
            H[i+1:, i+1:] = self.make_householder(A[i+1:, i])
            Q = np.dot(Q, H)
            A = np.dot(H, A)
            A = np.dot(A,H)
        return Q, A

    def qlnr(self, d,e,z,tol = 1.0e-9):
        #d - diagonal values
        #e - off-tridiag values
        #z - orthogonal matrix to process further
        n=len(d)
        e=np.roll(e,-1) #reorder
        itmax=1000
        for l in range(n):
            for iter in range(itmax):
                m=n-1
                for mm in range(l,n-1):
                    dd=abs(d[mm])+abs(d[mm+1])
                    if abs(e[mm])+dd == dd:
                        m=mm
                        break
                    if abs(e[mm]) < tol:
                        m=mm
                        break
                if iter==itmax-1:
                    print ("too many iterations",iter)
                    # sys.exit(0)
                    break
                if m!=l:
                    g=(d[l+1]-d[l])/(2.*e[l])
                    r=np.sqrt(g*g+1.)
                    g=d[m]-d[l]+e[l]/(g+np.sign(g)*r)
                    s=1.
                    c=1.
                    p=0.
                    for i in range(m-1,l-1,-1):
                        f=s*e[i]
                        b=c*e[i]
                        if abs(f) > abs(g):
                            c=g/f
                            r=np.sqrt(c*c+1.)
                            e[i+1]=f*r
                            s=1./r
                            c *= s
                        else:
                            s=f/g
                            r=np.sqrt(s*s+1.)
                            e[i+1]=g*r
                            c=1./r
                            s *= c
                        g=d[i+1]-p
                        r=(d[i]-g)*s+2.*c*b
                        p=s*r
                        d[i+1]=g+p
                        g=c*r-b
                        for k in range(n):
                            f=z[k,i+1]
                            z[k,i+1]=s*z[k,i]+c*f
                            z[k,i]=c*z[k,i]-s*f
                    d[l] -= p
                    e[l]=g
                    e[m]=0.
                else:
                    break
        return d,z

    def QR_trid(self, matrix):
        Q, Trid = self.trid_householder(matrix)
        n=Trid.shape[0]
        d=np.zeros(n)
        e=np.zeros(n)
        for i in range(n):
            d[i]=Trid[i,i]
        for i in range(n-1):
            e[i+1]=Trid[i+1,i]
        
        return self.qlnr(d,e,Q)[0]
    def QR(self, matrix):
        eps = 1e-3
        def offDiag(A:np.array) -> float:
            return np.sum(A) - np.trace(A)
        A = np.copy(matrix)
        while abs(offDiag(A)) > eps:
            q,r = self.qr(A)
            A = r @ q
        return 0
    
    def use_alg(self, k):
        return self.algorithms[k](self.matrix)