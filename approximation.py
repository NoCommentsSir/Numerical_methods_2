import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from scipy.misc import derivative
from scipy.special import factorial
from scipy import integrate
import sympy as sym

def f(x):
    return x*np.tan(x)

def Chebishev_nodes(a, b, n):
    arr = []
    for i in range(n):
        x = 0.5*((b-a)*np.cos((2*i+1)*np.pi/(2*n+2))+b+a)
        arr.append(x)
    return np.array(arr)

def sq_polinom(x, y):
    Q = []
    for i in range(len(x)):
        temp = []
        for j in range(4):
            temp.append(x[i]**j)
        Q.append(temp)
    Q = np.asarray(Q)
    Q_T = Q.transpose()
    H = np.matmul(Q_T, Q)
    b = np.matmul(Q_T, y)
    a = np.linalg.solve(H, b)
    return a[::-1]

def Lagr_param(x_arr, y_arr):
    coef = []
    for i in range(len(x_arr)):
        temp = 1
        for j in range(len(x_arr)):
            if i != j:
                temp *= x_arr[i] - x_arr[j]
        coef.append(y_arr[i]/temp)
    return coef

def Lagr(param, nodes, x):
    result = 0
    for i in range(len(nodes)):
        temp = 1
        for j in range(len(nodes)):
            if i != j:
                temp *= x - nodes[j]
        result += temp*param[i]
    return result

def Lezhandr():
    p = sym.Symbol('p')
    H = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            g = 1/(factorial(j)*(2**j))
            H[i][j] = sym.integrate(g*sym.diff((1-p**2)**j,p, j)*g*sym.diff((1-p**2)**i,p, i), (p, -1, 1))
    b = np.zeros(4)
    for i in range(len(b)):
        b[i] = sym.integrate(p*sym.tan(p)*g*sym.diff((1-p**2)**i,p, i), (p, -1, 1))
    a = np.linalg.solve(H, b)
    return a[::-1]

def E(y1, y2, dx):
    return abs(y1-y2)*dx

print("Задание 1")
x = np.arange(-1,1.1,0.1)
y = f(x)
print("Число узлов | Отклонение по равноотстоящим | Отклонение по Чебышеву")
for n in range(3, 11):
    y_1 = []
    length = round(2 / n, 2)
    nodes = np.arange(-1, 1.1, length)
    means = f(nodes)
    a = sq_polinom(nodes, means)
    y_1 = np.polyval(a, x)
    cheb_nodes = Chebishev_nodes(-1, 1, n)
    cheb_means = f(nodes)
    a_cheb = sq_polinom(nodes, means)
    y_2 = np.polyval(a_cheb, x)
    sq_polinom(nodes, means)
    e_1 = max(E(y_1, y, length))
    e_2 = max(E(y_2, y, length))
    print(f'{n} | {e_1} | {e_2}')
    if n == 4:
        p_1 = y_1
        p_2 = y_2
print("===================================================================")
print("Задание 2")
print("Таблица")
for n in range(4, 11):
    print(f'{n} узлов')
    length = round(2.3 / n, 2)
    nodes = np.arange(-1, 1.1, length)
    q = np.arange(-5,5,len(range(-5,5))/n)
    means_1 = f(nodes)
    means = []
    for i in range(len(nodes)):
        means.append(f(nodes[i])*(100-q[i])/100)
    means = np.array(means)
    for i in range(len(means)):
        print(f'x: {nodes[i]},  y: {f(nodes[i])},  y_: {means[i]}')
    a = sq_polinom(nodes, means)
    y_1 = np.polyval(a, x)
    coef = Lagr_param(nodes, means)
    lagr_y = Lagr(coef, nodes, x)
    coef_1 = Lagr_param(nodes, means_1)
    lagr_y_1 = Lagr(coef_1, nodes, x)
    e_1 = max(E(y_1, y, length))
    e_2 = max(E(lagr_y, y, length))
    e_3 = max(E(lagr_y_1, y, length))
    print(f'Погрешности на {n} узлов | обобщ. полином: {e_1} | Лагранж: {e_2}')
    print(f'Погрешности на полинома Лагранжа на точных узлах: {e_3} | на приближенных: {e_2}')
    if n == 4:
        q_1 = y_1
        q_2 = lagr_y
print("===================================================================")
print("Задание 3")
for n in range(4, 11):
    length = round(2.3 / n, 2)
    nodes = np.arange(-1, 1.1, length)
    means = f(nodes)
    nodes_1 = []
    for k in nodes:
        nodes_1 += [k]*3
    means_1 = []
    for i in means:
        means_1 += [i*0.95, i, i*1.05]
    a = sq_polinom(nodes_1, means_1)
    y_1 = np.polyval(a, x)
    if n == 4:
        g_1 = y_1
print("===================================================================")
print("Задание 4")
length = round(2.3 / 6, 2)
nodes = np.arange(-1, 1.1, length)
means = f(nodes)
a = Lezhandr()
z_1 = np.polyval(a, x)
e_1 = max(E(z_1, y, length))
print(f'{6} | {e_1}')
fig, axis = plt.subplots(2,2)
axis[0,0].plot(x, y, linewidth=2.0)
axis[0,0].set_title("Задание 1")
axis[0,0].plot(x, p_1, color="orange", linewidth=3.0)
axis[0,0].plot(x, p_2, color="green", linewidth=1.0)
axis[0,0].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[0,0].grid()
axis[1,0].plot(x, y, linewidth=2.0)
axis[1,0].set_title("Задание 2")
axis[1,0].plot(x, q_1, color="orange", linewidth=3.0)
axis[1,0].plot(x, q_2, color="green", linewidth=1.0)
axis[1,0].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[1,0].grid()
axis[0,1].plot(x, y, linewidth=2.0)
axis[0,1].set_title("Задание 3")
axis[0,1].plot(x, g_1, color="orange", linewidth=3.0)
axis[0,1].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[0,1].grid()
axis[1,1].plot(x, y, linewidth=2.0)
axis[1,1].set_title("Задание 4")
axis[1,1].plot(x, z_1, color="orange", linewidth=3.0)
axis[1,1].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[1,1].grid()
plt.show()