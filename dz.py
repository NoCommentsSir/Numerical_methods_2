import numpy as np
import time
import copy
from numpy import linalg as la
from numpy.linalg import norm
np.set_printoptions(precision=7)
n = int(input())
eps = 1e-6
delta = 1e-8
def create_matrix(n):
    C = np.random.randint(-10, 10, (n, n))
    inv_C = la.inv(C)
    help_vec = np.random.randint(-10, 10, (n, 1))
    A = np.eye(n)
    A = A*help_vec
    print(A)
    A = np.dot(inv_C, A)
    A = np.dot(A, C)
    return A, help_vec, inv_C, C

def findLU(A, n):  # функция LU-разложения
    a = np.array(A, dtype="float")
    L_ = np.eye(n, dtype="float")
    Q_ = np.eye(n)

    for k in range(n):
        for w in range(k + 1, n):
            if a[k, k] != 0:
                a[w, k] = a[w, k] / a[k, k]
            else:
                break
            for q in range(k + 1, n):
                a[w, q] -= a[k, q] * a[w, k]

    for j in range(n):
        for i in range(j + 1, n):
            L_[i, j] = a[i, j]
            a[i, j] = 0

    U_ = a
    return L_, U_

def forwardSub(L, b, m):
    y = np.zeros_like(b, dtype="float")
    y[0] = b[0]
    for i in range(1, m):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def backSub(U, y, m):
    x = np.zeros_like(y, dtype="float")
    for i in range(m - 1, -1, -1):
        if (U[i, i] != 0):
            x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i, i]
        else:
            x[i] = 0
    return x

def uravn(A, n, b):
    L, U = findLU(A, n)
    y = forwardSub(L, b, n)
    x = backSub(U, y, n)
    return x

def inv(A, m):  # нахождение обратной матрицы
    L, U = findLU(A, m)
    b = np.eye(m)
    Ainv = np.zeros((m, m), dtype="float")
    for i in range(m):
        y = forwardSub(L, b[:, i], m)
        Ainv[:, i] = backSub(U, y, m)
    return Ainv

def deg_method(A, y_0):
    current_y = copy.copy(y_0)
    current_z = current_y*(1/norm(current_y, ord=2))
    next_y = np.dot(A, current_z)
    next_z = next_y*(1/norm(next_y, ord=2))
    cur_lambda = np.zeros(n)
    next_lambda = np.zeros(n)
    for i in range(len(next_y)):
        if abs(current_z[i]) > delta:
            next_lambda[i] = next_y[i]/current_z[i]
    while norm(next_lambda-cur_lambda, ord=1) > eps*max(norm(next_lambda,ord=1), norm(cur_lambda, ord=1)):
        cur_lambda = copy.copy(next_lambda)
        current_z = copy.copy(next_z)
        next_y = np.dot(A, current_z)
        next_z = next_y*(1/norm(next_y, ord=2))
        I = 0
        for i in range(len(next_y)):
            if abs(current_z[i]) > delta:
                next_lambda[i] = next_y[i] / current_z[i]
                I += 1
    lambda_1 = 0
    for i in range(len(next_lambda)):
        lambda_1 += next_lambda[i]
    lambda_1 /= I
    return lambda_1, next_z

def rev_deg_metod(A, y_0, sigma, const=True):
    start = time.time()
    count_iterations = 0
    current_sigma = sigma
    next_sigma = current_sigma
    lambda_0 = 0
    lambda_1 = 0
    current_y = copy.copy(y_0)
    current_z = current_y * (1 / norm(current_y, ord=2))
    cond = False
    I = np.eye(n)
    B = A - current_sigma * I
    L, U = findLU(B, n)
    while not cond:
        if count_iterations > 10000:
            print("Iteartions > 10000")
            break
        count_iterations += 1
        lambda_vec = np.array([0 for x in range(n)], dtype="float")

        if const == False:
            I = np.eye(n)
            B = A - next_sigma*I
            L, U = findLU(B, n)

        q = forwardSub(L, current_z, n)
        next_y = backSub(U, q, n)
        next_z = next_y*(1/norm(next_y, ord=2))
        I = 0
        for i in range(len(next_y)):
            if abs(current_y[i]) > delta:
                lambda_vec[i] = current_z[i] / next_y[i]
                I += 1
        lambda_1 = sum(lambda_vec[i] for i in range(n))/I

        if const == True:
            cond1 = abs(lambda_1 - lambda_0) < eps
            cond2 = norm(abs(next_z) - abs(current_z), ord=2) < eps
            cond = cond1 and cond2
        else:
            next_sigma = current_sigma + lambda_1
            cond1 = abs(next_sigma - current_sigma) < eps
            cond2 = norm(next_z - current_z, ord=2) < eps
            cond = cond1 and cond2
            current_sigma = copy.copy(next_sigma)
        current_z = copy.copy(next_z)
        lambda_0 = copy.copy(lambda_1)
    if const == True:
        return lambda_1 + sigma, next_z, count_iterations, start
    else:
        return next_sigma, next_z, count_iterations, start

def sgn_plus(i):
    if i >= 0:
        return 1
    else:
        return -1

def H_construct(A, n, p, shift):
    I = np.eye(n)
    summ = 0
    for i in range(p+shift, n):
        summ += A[i][p] ** 2
    summ = summ ** 0.5
    s = -1*sgn_plus(A[p+shift][p])*summ
    mu = 1/abs(2*s*(s-A[p+shift][p]))**0.5
    arr = []
    for k in range(n):
        if k < p+shift:
            arr.append(0)
        elif k == p + shift:
            arr.append(A[k][p] - s)
        else:
            arr.append(A[k][p])
    v = np.array(arr)
    v = mu*v
    v = v.reshape(n,1)
    return  I - 2*np.dot(v, v.reshape(1, n))

def qr_decompose(A, n):
    B = copy.copy(A)
    H = np.eye(n, dtype='float')
    for k in range(n-1):
        current_H = H_construct(B,n, k, 0)
        B = np.dot(current_H, B)
        H = np.dot(H, current_H)
    Q = H
    R = np.dot(H.transpose(), A)
    return Q, R

def disp(A):
    B = copy.copy(A)
    for i in range(n):
        for j in range(n):
            B[i][j] = round(B[i][j], 5)
    print(B)

def Hessenberg(A):
    B = copy.copy(A)
    for k in range(n - 2):
        current_H = H_construct(B, n, k, 1)
        C = np.dot(current_H, B)
        B = np.dot(C, current_H)
    return B

def QR_shift(A, n):
    array = []
    current_bn = None
    shift_matr = copy.copy(A)
    while n > 0:
        I = np.eye(n)
        bn = shift_matr[n-1][n-1]
        if n > 1:
            bn_1 = shift_matr[n-1][n-2]
        else:
            bn_1 = None
        shift_matr = shift_matr - bn*I
        Q, R = qr_decompose(shift_matr, n)
        shift_matr = np.dot(R, Q) + bn*I
        if (True if bn_1 is None else abs(bn_1) < eps) and (False if current_bn is None else abs(bn - current_bn) < abs(current_bn / 3)):
            array.append(bn)
            shift_matr = shift_matr[:n - 1, :n - 1]
            current_bn = None
            n -= 1
        else:
            current_bn = bn
    return array

def fix_A(diag, inv_C, C, n):
    fixA = np.eye(n)
    fixA = fixA * diag
    x = copy.copy(fixA[n-1][n-1])
    fixA[n-1][n-1] = fixA[n-2][n-2]
    fixA[n-1][n-2] = x
    fixA[n-2][n-1] = -x
    fixA = np.dot(inv_C, fixA)
    fixA = np.dot(fixA, C)
    return fixA, diag

def QR(A, n):
    bn = np.diag(A)
    matrix = A
    previous_bn = None
    while True:
        bn_1 = np.diagonal(matrix, offset=-1)
        Q, R = qr_decompose(matrix, n)
        matrix = np.dot(R, Q)
        flg1 = False
        flg2 = False
        for i in range(len(bn_1)):
            if abs(bn_1[i]) < delta:
                flg1 = True
            else:
                flg1 = False
                break
        if previous_bn is None:
            flg2 = False
        else:
            for i in range(len(bn_1)):
                if abs(bn[i] - previous_bn[i]) < abs(previous_bn[i] / 3):
                    flg2 = True
                else:
                    flg2 = False
                    break
        if flg1 and flg2:
            answer = np.diag(matrix)
            return answer

        array = []
        for i in range(n - 1):
            if abs(bn_1[i]) > delta:
                array.append(i)

        if len(array) == 1 and (False if previous_bn is None else (abs(bn - previous_bn) < abs(previous_bn/3)).all()):
            print("Матрица с блоком 2х2:")
            print(matrix)
            i = array[0]
            block = matrix[i:i+2, i:i+2]
            print("Блок 2х2:")
            print(block)

            help_matrix = block
            current_root = None
            previous_root = None
            root_diff = np.array([1e6, 1e6])
            while root_diff[0] > delta or root_diff[1] > delta:
                QQ, RR = qr_decompose(help_matrix, 2)
                help_matrix = np.dot(RR, QQ)
                a = 1
                b = -help_matrix[0][0]-help_matrix[1][1]
                c = help_matrix[0][0]*help_matrix[1][1]-help_matrix[0][1]*help_matrix[1][0]
                D = np.complex_(b ** 2 - 4 * a * c + 0j)
                arr = []
                arr.append(np.complex_((-b + np.sqrt(D, dtype='complex_')) / 2))
                arr.append(np.complex_((-b - np.sqrt(D, dtype='complex_')) / 2))
                current_root = np.array(arr)
                if previous_root is not None:
                    root_diff = np.abs(current_root - previous_root)
                previous_root = current_root
            print(current_root)
            answer = np.complex_(copy.copy(np.diag(matrix)))
            answer[i] = current_root[0]
            answer[i+1] = current_root[1]
            return answer



        previous_bn = bn

print("Собственные значения:")
A, diag, inv_C, C = create_matrix(n)
y_0 = np.random.randint(-10, 10, (n, 1))
maxx = -10**8
for i in range(n):
    if abs(diag[i]) > maxx:
        maxx = abs(diag[i])
        q = sgn_plus(diag[i])
maxx = q*maxx
Hess = Hessenberg(A)
print("Матрица А:")
disp(A)

print("Матрица А в форме Хессенберга:")
disp(Hess)
print('=================================================================')
value_2, vector_2 = deg_method(A, y_0)
print(maxx, "- собственное число; ", "cобственное число из алгоритма:", value_2, "; \nCобственный вектор из алгоритма:\n", vector_2,)
print('Проверка:')
Ax_2 = np.dot(A, vector_2)
lx_2 = value_2*vector_2
print("Ax:", Ax_2, "\nlx:", lx_2)
print()
print('=================================================================')
print("QR - алгоритм со сдвигами:")
lambda_vec = QR_shift(Hess, n)
print(lambda_vec)
print('=================================================================')
for i in range(len(lambda_vec)):
    value_1, vector_1, iterations_1, start_time_1 = rev_deg_metod(A, y_0, lambda_vec[i], const=True)
    print(diag[i], "- текущее собственное число; ", "сдвиг:",  lambda_vec[i],  "cобственное число из алгоритма:", value_1, "; \nCобственный вектор из алгоритма:\n", vector_1,
          ". \nВсего итераций:", iterations_1, ", затраченное время:", time.time() - start_time_1)
    value, vector, iterations, start_time = rev_deg_metod(A, y_0, lambda_vec[i], const=False)
    print(diag[i], "- текущее собственное число; ", "сдвиг:",  lambda_vec[i], "cобственное число из алгоритма:", value,
          "; \nCобственный вектор из алгоритма:\n", vector,
          ". \nВсего итераций:", iterations, ", затраченное время:", time.time() - start_time)
    print('Проверка:')
    Ax = np.dot(A, vector)
    lx = value* vector
    print("Ax:", Ax, "\nlx:", lx, "\nAx=lx:", norm(Ax - lx, ord=2) < eps)
    print()
print('=================================================================')
A_1, new_diag = fix_A(diag, inv_C, C, n)
print("Новая матрица А:")
print(A_1)
print(QR(A_1, n))
