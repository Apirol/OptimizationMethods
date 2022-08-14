import math
import random

import numpy as np
import numpy.linalg as ln
import scipy.optimize

from method import golden_ratio, calculate_vector_sum, calculate_x
import numdifftools as nd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, golden


def fast_gradient_method(f, fprime, x0, maxiter=10000, epsi=1e-6):
    df = pd.DataFrame()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    X = np.arange(-20, 20, 0.5)
    Y = np.arange(-20, 20, 0.5)
    X, Y = np.meshgrid(X, Y)
    plt.contour(X, Y, f([X, Y]))
    plt.xlabel('Current X')
    plt.ylabel('Current Function')

    plt.plot(x0[0], x0[1], 'ro')

    iter_counter: int = 0
    xk = x0
    current_gradient = fprime(x0)
    iter_counter_func = 0

    while iter_counter < maxiter:
        q = lambda alpha: f(xk - alpha * current_gradient)
        coeff = golden(q)
        #iter_counter_func += iter_func
        xNext = xk - coeff * current_gradient

        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': coeff, 'DX': xNext[0] - xk[0],
                         'DY': xNext[1] - xk[1], 'DF': f(xNext) - f(xk), 'Gradient': current_gradient}]
        df = df.append(current_iter, ignore_index=True)

        plt.plot([xk[0], xNext[0]], [xk[1], xNext[1]], '-r')

        xk = xNext
        current_gradient = fprime(xk)
        iter_counter += 1

        if np.linalg.norm(current_gradient) <= epsi:
            return xk


def function(X):
    x, y = X
    return 5 * (x + y)**2 + (x - 2)**2


def func_for_simple(X):
    x, y = X
    #a = [-6, -7, 8, -9, 9, 0]
    #b = [9, 7, -8, 3, 8, 7]
    #C = [7, 9, 10, 6, 5, 7]
    a = [5, 2, -9, 0, -3, -3]
    b = [4, 0, -6, -3, 7, 3]
    C = [2, 1, 7, 2, 8, 4]
    res = 0

    for i in range(6):
        res += C[i] / (1 + (x - a[i])**2 + (y - b[i])**2)

    return -res


def gradient(x):
    grad = nd.Gradient(function)
    dx, dy = grad([x[0], x[1]])
    return np.array([dx, dy])


def g(X):
    x, y = X
    return 1 - x - y
    #return 2 - y + x


def h(X):
    x, y = X
    return abs(x - y)



def penalty_function(g):
    return 0.5 * (g + abs(g))


def barrier_function(g):
   #return -np.log(-g)
   return -1 / g


def penalty_method(f, gradient, penalty_function, g, x_0, eps=1e-5, maxiter=10000):
    df = pd.DataFrame()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    X = np.arange(-20, 20, 0.5)
    Y = np.arange(-20, 20, 0.5)
    X, Y = np.meshgrid(X, Y)
    plt.contour(X, Y, f([X, Y]))

    plt.plot(x_0[0], x_0[1], 'ro')

    iter_counter = 0
    w = 10
    rg = 10000
    rh = 1
    xk = x_0

    while iter_counter < maxiter:
        current_f = lambda x: f(x) + max(0, rg * penalty_function(g(x))**2 * w) + max(0, rh * h(x)**4 * w)
        xNext = minimize(current_f, xk).x

        xk = xNext
        current_value = current_f(xNext)

        temp = rh * h(xk) * w
        q = max(0, rg * penalty_function(g(xk))**2 * w) + rh * h(xk)**4 * w
        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Q': q, 'Current_iter': iter_counter}]
        df = df.append(current_iter, ignore_index=True)

        if q < eps:
            df.to_excel("Penalty.xlsx")
            return xk, iter_counter

        w *= 100
        iter_counter += 1


def barrier_method(f, gradient, penalty_function, g, x_0, eps=1e-5, maxiter=10000):
    df = pd.DataFrame()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    X = np.arange(-20, 20, 0.5)
    Y = np.arange(-20, 20, 0.5)
    X, Y = np.meshgrid(X, Y)
    plt.contour(X, Y, f([X, Y]))

    plt.plot(x_0[0], x_0[1], 'ro')

    iter_counter = 0
    w = 1
    coeff = 10E5
    xk = x_0

    while iter_counter < maxiter:
        temp = coeff * barrier_function(g(xk)) * w
        current_f = lambda x: f(x) + coeff * barrier_function(g(x)) * w
        xNext = minimize(current_f, xk, method='Nelder-Mead').x

        xk = xNext
        tem = g(xk)
        current_value = current_f(xNext)

        q = coeff * barrier_function(g(xk)) * w
        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Q': q, 'Current_iter': iter_counter}]
        df = df.append(current_iter, ignore_index=True)

        if abs(q) < eps:
            df.to_excel("Barrier.xlsx")
            return xk, iter_counter

        w *= 0.0001
        iter_counter += 1


def second_Pearson(f, fprime, x0, maxiter=10000, epsi=1e-4):
    df = pd.DataFrame()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    X = np.arange(-20, 20, 0.5)
    Y = np.arange(-20, 20, 0.5)
    X, Y = np.meshgrid(X, Y)
    plt.contour(X, Y, f([X, Y]))

    plt.plot(x0[0], x0[1], 'ro')

    k = 0
    current_gradient = fprime(x0)
    N = len(x0)

    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    alpha_k = 0.005
    iter_counter_func = 0


    while k < maxiter:
        pk = np.dot(Hk, current_gradient)
        q = lambda alpha: f(xk - alpha * pk)
        alpha_k, iter_func = golden_ratio(q, 0, 10, epsi)
        iter_counter_func += iter_func

        xkp1 = xk - alpha_k * pk

        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': alpha_k, 'DX': xkp1[0] - xk[0],
                         'DY': xkp1[1] - xk[1], 'DF': f(xkp1) - f(xk), 'Gradient': current_gradient, 'Hk': Hk}]
        df = df.append(current_iter, ignore_index=True)
        plt.plot([xk[0], xkp1[0]], [xk[1], xkp1[1]], '-r')

        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - current_gradient
        current_gradient = gfkp1
        k += 1

        temp = 1 / np.dot(yk[np.newaxis, :], sk)

        if k % 2 == 0:
            Hk = np.eye(N, dtype=int)
        else:
            Hk = Hk + ((np.dot(sk - np.dot(Hk, yk), np.dot(yk, Hk).transpose())) * temp)

        if ln.norm(current_gradient) < epsi:
            df.to_excel("Report_Second_Pearson.xlsx")
            plt.show()
            return xk, k, iter_counter_func

    df.to_excel("Report_Second_Pearson.xlsx")
    plt.show()
    return xk, k, iter_counter_func


def third_Pearson(f, fprime, x0, maxiter=10000, epsi=1e-4):
    df = pd.DataFrame()

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    X = np.arange(-20, 20, 0.5)
    Y = np.arange(-20, 20, 0.5)
    X, Y = np.meshgrid(X, Y)
    plt.contour(X, Y, f([X, Y]))

    plt.plot(x0[0], x0[1], 'ro')

    k = 0
    current_gradient = fprime(x0)
    N = len(x0)

    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    alpha_k = 0.005
    iter_counter_func = 0

    while k < maxiter:
        pk = np.dot(Hk, current_gradient)
        q = lambda alpha: f(xk - alpha * pk)
        alpha_k, iter_func = golden_ratio(q, 0, 1, epsi)
        iter_counter_func += iter_func
        xkp1 = xk - alpha_k * pk

        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': alpha_k, 'DX': xkp1[0] - xk[0],
                         'DY': xkp1[1] - xk[1], 'DF': f(xkp1) - f(xk), 'Gradient': current_gradient, 'Hk': Hk}]
        df = df.append(current_iter, ignore_index=True)

        plt.plot([xk[0], xkp1[0]], [xk[1], xkp1[1]], '-r')

        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - current_gradient
        current_gradient = gfkp1
        k += 1

        temp = 1 / np.dot(np.dot(yk[np.newaxis, :], Hk), yk)

        if k % 22 == 0:
            Hk = np.eye(N, dtype=int)
        else:
            Hk = Hk + ((np.dot(sk - np.dot(Hk, yk), np.dot(yk, Hk).transpose())) * temp)

        if ln.norm(current_gradient) < epsi:
            return xk

    df.to_excel("Report_Third_Pearson.xlsx")
    plt.show()
    return xk, k, iter_counter_func


def simple_search(function, P, eps):
    V_x = [-10, 10]
    V_y = V_x

    V = (V_x[1] - V_x[0]) * (V_y[1] - V_y[0])

    V_eps = eps * eps
    P_eps = V_eps / V
    N = round(np.log(1 - P) / np.log(1 - P_eps))
    min_x = [0, 0]
    global_min = float('inf')

    iter_counter = 0
    while iter_counter < N:
        first_x = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
        second_x = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])

        if function(first_x) < function(second_x) and function(first_x) < global_min:
            min_x = first_x
            global_min = function(min_x)
        elif function(first_x) > function(second_x) and function(second_x) < global_min:
            min_x = second_x
            global_min = function(min_x)
        iter_counter += 1
    return min_x, N


def StohatisticGradientMethod(function, eps, g, m, maxiter=10000):
    V_x = [-10, 10]
    V_y = V_x

    iter_counter: int = 0
    alpha = 1
    xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])

    while iter_counter < maxiter:
        delta_f = calculate_vector_sum(function, xk, m, g, function(xk), V_x, V_y)

        q = lambda alpha: function(xk - alpha * delta_f)
        coeff = golden(q)
            #golden_ratio(q, -100, 100, eps)
        xNext = xk + coeff * delta_f / ln.norm(delta_f)

        if ln.norm(delta_f) < eps:
            return xNext

        xk = xNext
        iter_counter += 1



def pair_method(function,eps, g, maxiter=100000):
    V_x = [-10, 10]
    V_y = V_x

    alpha = 10
    iter_counter: int = 0
    xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])

    while iter_counter < maxiter:
        first_x = calculate_x(xk, V_x, V_y)

        first_func = function(xk + g * first_x)
        second_func = function(xk - g * first_x)

        if first_func < second_func:
            xk = xk + alpha * first_x * first_func
        elif second_func < first_func:
            xk = xk + alpha * first_x * second_func

        iter_counter += 1

    return xk


def Alg1(function, M, sd):
    V_x = [-10, 10]
    V_y = V_x

    random.seed(sd)
    iter_counter: int = 0
    func_counter = 0
    m = 0
    xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
    current_value = 0
    fmin = float('inf')
    xmin = 0

    while m < M:
        iter_counter += 1
        xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])

        xNext = minimize(function, xk, method='Nelder-Mead').x
        current_func = function(xNext)
        func_counter += 1

        if current_func < fmin:
            fmin = current_func
            xmin = xNext
        m += 1

    return xmin, fmin, func_counter


def Alg2(function, M, sd):
    V_x = [-10, 10]
    V_y = V_x

    random.seed(sd)
    iter_counter: int = 0
    func_counter = 0
    m = 0
    xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
    xmin = minimize(function, xk, method='Nelder-Mead').x
    fmin = function(xmin)

    while True:
        iter_counter += 1
        m = 0
        while m < M:
            xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])

            current_func = function(xk)
            func_counter += 1

            if current_func < fmin:
                fmin = current_func
                xmin = xk
                break
            m += 1
        if m == M:
            return xmin, fmin, func_counter
        else:
            xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
            xmin = minimize(function, xk, method='Nelder-Mead').x
            fmin = function(xmin)
            func_counter += 1


def Alg3(function, M, sd):
    V_x = [-10, 10]
    V_y = V_x

    random.seed(sd)
    iter_counter: int = 0
    func_counter = 0
    m = 0
    xk = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
    xmin = minimize(function, xk, method='Nelder-Mead').x
    fmin = function(xmin)
    delta = 0.5

    while True:
        iter_counter += 1
        m = 0
        while m < M:
            m += 1
            gen_x = np.array([random.uniform(V_x[0], V_x[1]), random.uniform(V_y[0], V_y[1])])
            point = [0, 0]
            while True:
                point[0] += delta * (gen_x[0] / math.sqrt(gen_x[0]**2 + gen_x[1]**2))
                point[1] += delta * (gen_x[1] / math.sqrt(gen_x[0] ** 2 + gen_x[1] ** 2))

                func_counter += 2
                if function(point) < function(xmin) or (point[0] < V_x[0] or point[0] > V_x[1] or point[1] < V_y[0] or point[1] > V_y[1]):
                    break

            if point[0] >= V_x[0] and point[0] <= V_x[1] and  point[1] >= V_y[0] and point[1] <= V_y[1] :
                break
        if m == M:
            return xmin, fmin, func_counter
        else:
            current_x = minimize(function, point, method='Nelder-Mead').x

            if function(current_x) < function(xmin):
                xmin = current_x
                func_counter += 1
                fmin = function(xmin)


test_func = func_for_simple([-3, 7])
P = [0.8, 0.9, 0.95, 0.99]
eps = [1, 1e-1]
"""
df = pd.DataFrame()
for i in range(len(eps)):
    for j in range(len(P)):
        min_x, N = simple_search(func_for_simple, P[j], eps[i])
        current_iter = [{'Eps': eps[i], 'P': P[j], 'X': min_x, 'N': N}]
        df = df.append(current_iter, ignore_index=True)
df.to_excel("simple_search.xlsx")
"""
df = pd.DataFrame()
m = [100, 500, 1000, 5000, 10000]
sd = [0, 100, -100, -456321, 1234568]
for i in range(len(sd)):
    min_x1, fmin1, func_counter1 = Alg1(func_for_simple, 1000, sd[i])
    min_x2, fmin2, func_counter2 = Alg2(func_for_simple, 1000, sd[i])
    min_x3, fmin3, func_counter3= Alg3(func_for_simple, 1000, sd[i])
    current_iter = [{'sd': sd[i], 'Alg1 X': min_x1, 'Alg2 X': min_x2, 'Alg3 X': min_x3, 'Function Alg1': fmin1,
                     'Function Alg2': fmin2, 'Function Alg3': fmin3, 'Alg1 Count': func_counter1, 'Alg2 Count': func_counter2,
                     'Alg3 Count': func_counter3}]
    df = df.append(current_iter, ignore_index=True)
df.to_excel("alg.xlsx")



global_min, min_x = simple_search(func_for_simple, 0.95, 1e-1)
global_min, min_x = pair_method(func_for_simple, 1e-1, 0.001)
x_min, f_min = Alg3(func_for_simple, 1e-3)
result = Alg1(func_for_simple, 1, 0.1)
print(min_x)
