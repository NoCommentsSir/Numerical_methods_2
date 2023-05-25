import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.misc import derivative

def f(x):
    return np.tan(x) - np.cos(3*x) + 0.1

def der_f(x):
    return 1/(np.cos(x)**2) + 3*np.sin(3*x)

def derr_f(x, dx = 1e-10):
    return (der_f(x+dx) - der_f(x))/dx

def h(x):
    return f(x)*abs(x)

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

def Chebishev_nodes(a, b, n):
    arr = []
    for i in range(n):
        x = 0.5*((b-a)*np.cos((2*i+1)*np.pi/(2*n+2))+b+a)
        arr.append(x)
    return np.array(arr)

def Chebishev_param(x_arr, flg=True):
    y_arr = []
    for i in range(len(x_arr)):
        if flg:
            y_arr.append(f(x_arr[i]))
        else:
            y_arr.append(h(x_arr[i]))
    params = Lagr_param(x_arr, y_arr)
    return params, y_arr

def Hermit_coef(nodes, k):
    matrix = []
    matrix1 = []
    m = len(nodes)*(k+1)-1
    for i in range(len(nodes)):
        temp = []
        for j in range(m+1):
            temp.append(nodes[i]**j)
        matrix.append(temp)
    for i in range(len(nodes)):
        for g in range(1,k+1):
            temp = [0]*g
            for j in range(m+1-g):
                if g == 1:
                    temp.append((j+g)*nodes[i]**j)
                if g == 2:
                    temp.append((j + g)*(j + g - 1) * nodes[i] ** j)
            matrix1.append(temp)
    for i in matrix1:
        matrix.append(i)
    y = f(nodes)
    if k == 1:
        y = np.append(y, der_f(nodes), axis=0)
    elif k == 2:
        for i in range(len(nodes)):
            y = np.append(y, [der_f(nodes[i]), derr_f(nodes[i])], axis=0)
    z = np.linalg.solve(np.asarray(matrix), y)
    return z

def Hermit(x, koef):
    arr = []
    for j in x:
        temp = []
        for i in range(len(koef)):
            temp.append(j**i)
        y = np.dot(np.array(temp), koef)
        arr.append(y)
    return np.array(arr)

def der_Herm(x, coef, dx=1e-10):
    return (Hermit(x + dx, coef) - Hermit(x, coef))/dx

def derr_Herm(x, coef, dx=1e-10):
    return (Hermit(x + 2*dx, coef) - Hermit(x+dx, coef))/dx

def der_Lagr(coef, nodes, x, dx=1e-10):
    return (Lagr(coef, nodes, x + dx) - Lagr(coef, nodes, x))/dx

def derr_Lagr(coef, nodes, x, dx=1e-6):
    return (der_Lagr(coef, nodes, x + dx) - der_Lagr(coef, nodes, x))/dx

def E(y1, y2, dx):
    return abs(y1-y2)*dx

def spline_10(x, nodes, means, length):
    arr = []
    for j in range(len(x)):
        for i in range(len(nodes)-1):
            if x[j] < nodes[i+1] and x[j] >= nodes[i]:
                arr.append((nodes[i+1] - x[j])*means[i]/length + (-nodes[i] + x[j])*means[i+1]/length)
    return np.array(arr)

def spline_31(x, nodes, means, der_means, length):
    arr = []
    for j in range(len(x)):
        for i in range(len(nodes)-1):
            if x[j] < nodes[i+1] and x[j] >= nodes[i]:
                t = (x[j] - nodes[i])/length
                alpha = (1 - t)**2*(1 + 2 * t)
                beta = t**2*(3 - 2*t)
                gamma = t*(1-t)**2
                delta = -t**2*(1-t)
                arr.append(alpha*means[i] + beta*means[i+1] + gamma*length*der_means[i] + delta*length*der_means[i+1])
    return np.array(arr)

def solver(lambd, mu, c, length):
    arr = np.zeros((length, length))
    coef = [2 for i in range(length)]
    for i in range(1, length+1):
        arr[i-1][i-2] = lambd[i-1]
        arr[i-1][i-1] = coef[i-1]
        try:
            arr[i-1][i] = mu[i-1]
        except:
            pass
    arr[0][-1] = 0
    c = c.transpose()
    m = np.linalg.solve(arr, c)
    return m

def spline_32(x, nodes, means, length):
    lambd = []
    mu = []
    c = []
    arr = []
    for i in range(1,len(nodes)-1):
        mu_1 = length/(2*length)
        lambd_1 = 1-mu_1
        c_1 = 3*(mu_1*(means[i+1] - means[i])/length + lambd_1*(means[i] - means[i-1])/length)
        mu.append(length/(2*length))
        lambd.append(lambd_1)
        c.append(c_1)
    m = solver(lambd, mu, np.array(c), len(range(1,len(nodes)-1)))
    m = np.append(m, der_f(nodes[-1])) # вставить 0 для естественного
    m = m.tolist()
    m.insert(0, der_f(nodes[0])) # вставить 0 для естественного
    for j in range(len(x)):
        for i in range(len(nodes)-1):
            if x[j] < nodes[i+1] and x[j] >= nodes[i]:
                t = (x[j] - nodes[i])/length
                alpha = (1 - t)**2*(1 + 2 * t)
                beta = t**2*(3 - 2*t)
                gamma = t*(1-t)**2
                delta = -t**2*(1-t)
                arr.append(alpha*means[i] + beta*means[i+1] + gamma*length*m[i] + delta*length*m[i+1])
    return np.array(arr)


print("Задание 1")
x = np.arange(-1,1.1,0.1)
y = f(x)
print("Число узлов | Отклонение по Лагранжу | Отклонение по Чебышеву")
for n in range(3, 13):
    length = round(2 / n, 2)
    nodes = np.arange(-1, 1.1, length)
    means = f(nodes)
    coef = Lagr_param(nodes, means)
    lagr_y = Lagr(coef, nodes, x)
    cheb_nodes = Chebishev_nodes(-1, 1, n)
    cheb_params, cheb_means = Chebishev_param(cheb_nodes)
    cheb_y = Lagr(cheb_params, cheb_nodes, x)
    e_lagr = max(E(lagr_y, y, length))
    e_cheb = max(E(cheb_y, y, length))
    print(f'{n} | {e_lagr} | {e_cheb}')
    if n == 3:
        n1 = nodes
        m1 = means
        l1 = lagr_y
        chn1 = cheb_nodes
        chm1 = cheb_means
        chl1 = cheb_y
x_2 = np.arange(-1,1.1,0.01)
y_2 = f(x_2)
lagr_y_2 = Lagr(coef, nodes, x)
cheb_nodes_2 = Chebishev_nodes(-1, 1, n)
cheb_params_2, cheb_means_2 = Chebishev_param(cheb_nodes_2)
cheb_y_2 = Lagr(cheb_params_2, cheb_nodes_2, x)
e_lagr = max(E(lagr_y_2, y, 0.01))
e_cheb = max(E(cheb_y_2, y, 0.01))
print(f'Для 100 точек: {n} | {e_lagr} | {e_cheb}')
print("===================================================================")
print("Задание 2")
x_h = np.arange(-1,1.1,0.1)
y_h = h(x_h)
print("Число узлов | Отклонение по Лагранжу | Отклонение по Чебышеву")
for n in range(3, 13):
    length = round(2/n,2)
    nodes = np.arange(-1, 1.1, length)
    means = h(nodes)
    coef = Lagr_param(nodes, means)
    lagr_y = Lagr(coef, nodes, x_h)
    cheb_nodes = Chebishev_nodes(-1, 1, n)
    cheb_params, cheb_means = Chebishev_param(cheb_nodes, False)
    cheb_y = Lagr(cheb_params, cheb_nodes, x_h)
    e_lagr = max(E(lagr_y, y_h, length))
    e_cheb = max(E(cheb_y, y_h, length))
    print(f'{n} | {e_lagr} | {e_cheb}')
    if n == 3:
        n1_h = nodes
        m1_h = means
        l1_h = lagr_y
        chn1_h = cheb_nodes
        chm1_h = cheb_means
        chl1_h = cheb_y
x_h_2 = np.arange(-1,1.1,0.01)
y_h_2 = f(x_2)
lagr_y_h_2 = Lagr(coef, nodes, x)
cheb_nodes_h_2 = Chebishev_nodes(-1, 1, n)
cheb_params_h_2, cheb_means_h_2 = Chebishev_param(cheb_nodes_h_2)
cheb_y_h_2 = Lagr(cheb_params_h_2, cheb_nodes_h_2, x_h)
e_lagr_h = max(E(lagr_y_h_2, y_h, 0.01))
e_cheb_h = max(E(cheb_y_h_2, y_h, 0.01))
print(f'Для 100 точек: {n} | {e_lagr_h} | {e_cheb_h}')
print("===================================================================")
print("Задание 3")
x_3 = np.arange(-1,1.1,0.1)
y_3 = f(x_3)
print("Число узлов | Отклонение по Эрмиту | Отклонение по Чебышеву | Отклонение по Лагранжу 1 | Отклонение по Лагранжу 2")
for n in range(3, 6):
    length = round(2 / n, 2)
    nodes = np.arange(-1, 1.1, length)
    means = f(nodes)
    length_2 = round(2 / (2*n - 1), 2)
    nodes_2 = np.arange(-1, 1.1, length_2)
    means_2 = f(nodes_2)
    coef = Lagr_param(nodes, means)
    lagr_y = Lagr(coef, nodes, x_3)
    coef_2 = Lagr_param(nodes_2, means_2)
    lagr_y_2 = Lagr(coef_2, nodes_2, x_3)
    a = Hermit_coef(nodes, 2)
    hermit_y = Hermit(x_3, a)
    cheb_nodes = Chebishev_nodes(-1, 1, n)
    cheb_means = f(cheb_nodes)
    a_cheb = Hermit_coef(cheb_nodes, 2)
    cheb_hermit_y = Hermit(x_3, a_cheb)
    e_herm = max(E(hermit_y, y_3, length))
    e_cheb = max(E(cheb_hermit_y, y_3, length))
    e_lagr = max(E(lagr_y, y_3, length))
    e_lagr_2 = max(E(lagr_y_2, y_3, length_2))
    e_herm_1 = max(E(der_Herm(x_3, a), der_f(x_3), length))
    e_cheb_1 = max(E(der_Herm(x_3, a_cheb), der_f(x_3), length))
    e_lagr_1 = max(E(der_Lagr(coef, nodes, x_3), der_f(x_3), length))
    e_lagr_21 = max(E(der_Lagr(coef_2, nodes_2, x_3), der_f(x_3), length_2))
    print(f'E: {n} | {e_herm} | {e_cheb} | {e_lagr} | {e_lagr_2}')
    print(f'E_: {n} | {e_herm_1} | {e_cheb_1} | {e_lagr_1} | {e_lagr_21}')
    if n == 3:
        n1_hermit = nodes
        m1_hermit = means
        l1_hermit = hermit_y
        chn1_hermit = cheb_nodes
        chm1_hermit = cheb_means
        chl1_hermit = cheb_hermit_y
print("===================================================================")
print("Задание 4")
print("Число узлов | Отклонение по Эрмиту | Отклонение по Чебышеву | Отклонение по Лагранжу 1 | Отклонение по Лагранжу 2")
length = round(2 / 3, 2)
nodes = np.arange(-1, 1.1, length)
means = f(nodes)
length_2 = round(2 / (2*3 - 1), 2)
nodes_2 = np.arange(-1, 1.1, length_2)
means_2 = f(nodes_2)
coef = Lagr_param(nodes, means)
lagr_y = Lagr(coef, nodes, x_3)
coef_2 = Lagr_param(nodes_2, means_2)
lagr_y_2 = Lagr(coef_2, nodes_2, x_3)
a = Hermit_coef(nodes, 2)
hermit_y = Hermit(x_3, a)
cheb_nodes = Chebishev_nodes(-1, 1, n)
cheb_means = f(cheb_nodes)
a_cheb = Hermit_coef(cheb_nodes, 2)
cheb_hermit_y = Hermit(x_3, a_cheb)
e_herm = max(E(hermit_y, y_3, length))
e_cheb = max(E(cheb_hermit_y, y_3, length))
e_lagr = max(E(lagr_y, y_3, length))
e_lagr_2 = max(E(lagr_y_2, y_3, length_2))
e_herm_1 = max(E(der_Herm(x_3, a), der_f(x_3), length))
e_cheb_1 = max(E(der_Herm(x_3, a_cheb), der_f(x_3), length))
e_lagr_1 = max(E(der_Lagr(coef, nodes, x_3), der_f(x_3), length))
e_lagr_21 = max(E(der_Lagr(coef_2, nodes_2, x_3), der_f(x_3), length_2))
e_herm_2 = max(E(derr_Herm(x_3, a), derr_f(x_3), length))
e_cheb_2 = max(E(derr_Herm(x_3, a_cheb), derr_f(x_3), length))
e_lagr_12 = max(E(derr_Lagr(coef, nodes, x_3), derr_f(x_3), length))
e_lagr_22 = max(E(derr_Lagr(coef_2, nodes_2, x_3), derr_f(x_3), length_2))
print(f'E: {3} | {e_herm} | {e_cheb} | {e_lagr} | {e_lagr_2}')
print(f'E_1: {3} | {e_herm_1} | {e_cheb_1} | {e_lagr_1} | {e_lagr_21}')
print(f'E_2: {3} | {e_herm_2} | {e_cheb_2} | {e_lagr_12} | {e_lagr_22}')
print("===================================================================")
print("Задание 5")
length = 2 / 4
nodes = np.arange(-1, 1.1, length)
means = f(nodes)
der_means = der_f(nodes)
spline1 = spline_10(x_3, nodes, means, length)
spline2 = spline_31(x_3, nodes, means, der_means, length)
spline3 = spline_32(x_3, nodes, means, length)
fig, axis = plt.subplots(2,2)
axis[0,0].plot(x, y, linewidth=2.0)
axis[0,0].set_title("Задание 1")
axis[0,0].plot(x, l1, color="orange")
axis[0,0].plot(x, chl1, color="green")
axis[0,0].scatter(n1, m1, color='orange', s=40, marker='o')
axis[0,0].scatter(chn1, chm1, color='green', s=40, marker='o')
axis[0,0].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[0,0].grid()
axis[0,1].plot(x_h, y_h, linewidth=2.0)
axis[0,1].set_title("Задание 2")
axis[0,1].plot(x_h, l1_h, color="orange")
axis[0,1].plot(x_h, chl1_h, color="green")
axis[0,1].scatter(n1_h, m1_h, color='orange', s=40, marker='o')
axis[0,1].scatter(chn1_h, chm1_h, color='green', s=40, marker='o')
axis[0,1].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[0,1].grid()
axis[1,0].plot(x_3, y_3, linewidth=2.0)
axis[1,0].set_title("Задание 3-4")
axis[1,0].plot(x_3, l1_hermit, color="orange")
axis[1,0].plot(x_3, chl1_hermit, color="green")
axis[1,0].scatter(n1_hermit, m1_hermit, color='orange', s=40, marker='o')
axis[1,0].scatter(chn1_hermit, chm1_hermit, color='green', s=40, marker='o')
axis[1,0].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[1,0].grid()
axis[1,1].plot(x_3, y_3, linewidth=2.0)
axis[1,1].set_title("Задание 5")
axis[1,1].plot(x_3, spline1, linewidth=2.0)
axis[1,1].plot(x_3, spline2, linewidth=2.0, color='green')
axis[1,1].plot(x_3, spline2, linewidth=1.0, color='red')
axis[1,1].scatter(nodes, means, color='orange', s=40, marker='o')
axis[1,1].set(xlim=(-1, 1), xticks=np.arange(-1, 1.1, 0.1),
       ylim=(-1, 1), yticks=np.arange(-1, 1.1, 0.1))
axis[1,1].grid()
plt.show()
