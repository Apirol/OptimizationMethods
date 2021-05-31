import numpy as np
import numpy.linalg as ln
from method import golden_ratio, search_minimal_segment
import numdifftools as nd
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen
import math


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
        coeff, iter_func = golden_ratio(q, -100, 100, epsi)
        iter_counter_func += iter_func
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
    #return x[0]**2 + x[1]**2
    x, y = X
    #return x**2 + y**2
    #return -x**2 - y**2
    return 5 * (x + y)**2 + (x - 2)**2
    #return (x - 4)**2 + (y - 4)**2
    #return 10 * (y - x)**2 + y**2
    #return x**2 - x*y + y**2 + 9*x - 6*y + 20
    #return 10 * x**2 + y**2
    #return x**2 + y**2 - 1.2 * x * y
    #return (1 - x) ** 2 + 100 * (y - x**2) ** 2
    #return -3 / (1 + (x[0] - 2)**2 + (x[1] - 2)**2 / 4) - 2 / (1 + (x[0] - 2)**2 / 9 + (x[1] - 3)**2)


def gradient(x):
    grad = nd.Gradient(function)
    dx, dy = grad([x[0], x[1]])
    return np.array([dx, dy])


def g(X):
    x, y = X
    return -x - y + 1


def h(X):
    x, y = X
    return -x - y + 1


def penalty_function(g):
    return 0.5 * (g + abs(g))


def barrier_function(g):
   return -1. / g


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
    r = 100
    xk = x_0

    while iter_counter < maxiter:
        current_f = lambda x, r: f(x) + max(0, r * penalty_function(g(x))**2 * w)
        xNext = minimize(current_f, xk, r).x
        xk = xNext
        current_value = current_f(xNext, r)

        q = r * penalty_function(g(xk))**2 * w
        if q < eps:
            return xk, iter_counter

        w *= 10
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
    w = 0.1
    r = 1000
    xk = x_0

    while iter_counter < maxiter:
        current_f = lambda x, r: f(x) + r * barrier_function(g(x)) * w + max(0, r * penalty_function(g(x))**2 * w) + r * h(xk) * w
        xNext = minimize(current_f, xk, r).x
        xk = xNext
        current_value = current_f(xNext, r)

        q = r * barrier_function(g(xk)) * w
        if abs(q) < eps:
            return xk, iter_counter

        w /= 10
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


#xk, counter, iter_counter = fast_gradient_method(function, gradient, np.array([10, 10]), epsi=1e-5)
#result, k, third_Pearson_iter = third_Pearson(function, gradient, np.array([10, 10]), epsi=1e-3)
#result1, k2, second_Pearson_iter = second_Pearson(function, gradient, np.array([10, 10]), epsi=1e-5)
xk, iter_counter = barrier_method(function, gradient, barrier_function, g, np.array([-10, 10]), eps=1e-5)
xk1, iter_counter1 = penalty_method(function, gradient, penalty_function, g, np.array([-10, 10]), eps=1e-5)
print("Stop")
