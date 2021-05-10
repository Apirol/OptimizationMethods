import numpy as np
import numpy.linalg as ln
from method import golden_ratio, search_minimal_segment
import numdifftools as nd
import pandas as pd
import matplotlib.pyplot as plt


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
        coeff, iter_func = golden_ratio(q, 0, 10, epsi)
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
            plt.show()
            df.to_excel('Report.xlsx')
            return xk, iter_counter, iter_counter_func

    plt.show()
    df.to_excel('Report.xlsx')
    return xk, iter_counter, iter_counter_func




def function(x):
    #return x[0]**2 + x[1]**2
    #x, y = X
    #return 10 * x**2 + y**2
    #return x**2-x*y+y**2+9*x-6*y+20
    #return (1 - x) ** 2 + 100 * (y - x**2) ** 2
    return -3 / (1 + (x[0] - 2)**2 + (x[1] - 2)**2 / 4) - 2 / (1 + (x[0] - 2)**2 / 9 + (x[1] - 3)**2)


def gradient(x):
    grad = nd.Gradient(function)
    dx, dy = grad([x[0], x[1]])
    return np.array([dx, dy])


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

        temp = 1 / np.dot(np.dot(yk[np.newaxis, :], Hk), yk)

        if k % 22 == 0:
            Hk = np.eye(N, dtype=int)
        else:
            Hk = Hk + ((np.dot(sk - np.dot(Hk, yk), np.dot(yk, Hk).transpose())) * temp)

        if ln.norm(current_gradient) < epsi:
            df.to_excel("Report_Third_Pearson.xlsx")
            plt.show()
            return xk, k, iter_counter_func

    df.to_excel("Report_Third_Pearson.xlsx")
    plt.show()
    return xk, k, iter_counter_func


xk, counter, iter_counter = fast_gradient_method(function, gradient, np.array([0, 0]), epsi=1e-5)
result, k, third_Pearson_iter = third_Pearson(function, gradient, np.array([0, 0]), epsi=1e-5)
result1, k2, second_Pearson_iter = second_Pearson(function, gradient, np.array([0, 0]), epsi=1e-5)
print("Stop")