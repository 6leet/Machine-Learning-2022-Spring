import numpy as np
from matplotlib import pyplot as plt

def parse(filename, n):
    file = open(filename, 'r')
    lines = file.readlines()
    A = []
    b = []
    for line in lines:
        x, y, _ = line.replace('\n', ',').split(',')
        x = float(x)
        y = float(y)
        t = 1
        a = [0] * n
        for i in range(n):
            a[i] = t
            t = t * x
        a.reverse()
        A.append(a)
        b.append(y)

    return A, b

class Matrix:
    def __init__(self, M, t=False):
        self.M = M
        self.t = t
        self.h = len(self.M)
        self.w = len(self.M[0])
        self.L = None
        self.U = None
        self.D = None

        if self.t:
            self.h, self.w = self.w, self.h

    def access(self, i, j):
        if self.t:
            i, j = j, i
        return self.M[i][j]

    def add(self, A):
        if self.h != A.h or self.w != A.w:
            return

        S = [[self.access(i, j) + A.access(i, j) for j in range(self.w)] for i in range(self.h)]
        return Matrix(S)

    def mult(self, a):
        self.M = [[em * a for em in m] for m in self.M]

        return self

    def product(self, A):
        if self.w != A.h:
            return

        S = [[0] * A.w for _ in range(self.h)]
        for i in range(self.h):
            for j in range(A.w):
                for k in range(self.w):
                    S[i][j] = S[i][j] + self.access(i, k) * A.access(k, j)

        return Matrix(S)

    def LUDecomposition(self):
        L = [[self.access(i, 0)] + [0] * (self.h - 1) for i in range(self.h)]
        U = [[int(i == j) for j in range(self.h)] for i in range(self.w)]

        for j in range(1, self.h):
            U[0][j] = self.access(0, j) / L[0][0]

        for j in range(1, self.h - 1):
            for i in range(j, self.h):
                L[i][j] = self.access(i, j) - sum(L[i][k] * U[k][j] for k in range(self.h - 1))
        
            for k in range(j + 1, self.h):
                U[j][k] = (self.access(j, k) - sum(L[j][i] * U[i][k] for i in range(self.h - 1))) / L[j][j]

        L[-1][-1] = self.access(-1, -1) - sum(L[-1][k] * U[k][-1] for k in range(self.h - 1))
        
        self.L = L
        self.U = U

    def forwardSubstitution(self, b):
        if self.L == None:
            self.LUDecomposition()

        D = [0] * self.h
        D[0] = b.access(0, 0) / self.L[0][0]
        for i in range(1, self.h):
            D[i] = (b.access(i, 0) - sum(self.L[i][j] * D[j] for j in range(i))) / self.L[i][i]

        self.D = D

    def __backSubstitution(self):
        X = [0] * self.h
        X[-1] = self.D[-1]

        for i in range(self.h - 1).__reversed__():
            X[i] = self.D[i] - sum(self.U[i][j] * X[j] for j in range(i + 1, self.h))
    
        return X

    def solve(self, b):
        self.forwardSubstitution(b)
        X = self.__backSubstitution()
        # print(X)

        return Matrix(X)

    def inverse(self):
        Inv = []
        for i in range(self.h):
            b = [[int(i == j) for j in range(self.h)]]
            self.forwardSubstitution(Matrix(b, t=True))
            Inv.append(self.__backSubstitution())

        return Matrix(Inv, t=True)

    def transpose(self):
        # self.t = not self.t
        # self.w, self.h = self.h, self.w
        # return self
        return Matrix(self.M, t=not self.t)

    @staticmethod
    def identity(n):
        return Matrix([[int(i == j) for j in range(n)] for i in range(n)])

    @staticmethod
    def LSE(A, b):
        At = A.transpose()
        AtA = At.product(A)
        return AtA.add(Matrix.identity(AtA.h).mult(lam)).inverse().product(At).product(b)

def fittingLine(x):
    fitLine = ''
    for i, r in zip(range(x.h), range(x.h).__reversed__()):
        if r == 0:
            fitLine += str(x.access(i, 0))
        else:
            fitLine += str(x.access(i, 0)) + 'X^' + str(r) + ' + '
    return fitLine

def error(est, b):
    return sum(pow(est.access(i, 0) - b.access(i, 0), 2) for i in range(b.h))

def plot(x, y, coeff):
    poly = np.poly1d(coeff)
    new_x = np.linspace(x[0] - 1, x[-1] + 1)
    new_y = poly(new_x)
    yticks = range(0, 120, 20)
    plt.xlim(x[0] - 1, x[-1] + 1)
    plt.yticks(yticks)
    plt.plot(x, y, "o", new_x, new_y)


filename = input('file name: ')
n = int(input('n: '))
lam = int(input('lambda: '))

A, b = parse(filename, n)
A = Matrix(A)
b = Matrix([b], t=True)
xp = [a[-2] for a in A.M]
yp = b.M[0]

# LSE
print('LSE:')
lse = Matrix.LSE(A, b)

## fitting line
print('Fitting line:', fittingLine(lse))

## error
est = A.product(lse)
print('Total error:', error(est, b))

## plot
plt.subplot(211)
coeff = [l[0] for l in lse.M]
plot(xp, yp, coeff)


# Newton
print('\nNewton\'s Method:')
AtA = A.transpose().product(A)
AtAInv = AtA.inverse()
x = Matrix([[0] * AtA.h], t=True)
est = A.product(x)
lastErr = error(est, b)
delta = 1e8
while delta > 1e-8:
    x = x.add(AtAInv.product(AtA).product(x).mult(-1)).add(AtAInv.product(A.transpose()).product(b))
    est = A.product(x)
    newErr = error(est, b)
    delta = abs(lastErr - newErr)
    lastErr = newErr

## fitting line
print('Fitting line:', fittingLine(x))

## error
print('Total error:', error(est, b))

## plot
plt.subplot(212)
coeff = [l[0] for l in x.M]
plot(xp, yp, coeff)

plt.show()