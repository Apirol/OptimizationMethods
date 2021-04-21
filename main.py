import numpy as np
import numpy.linalg as ln
from method import fibonacci,search_minimal_segment, search_minimal_segment2, fibonacci2
import numdifftools as nd
import pandas as pd



def function_alpha(alpha, xk):
    return function(xk - alpha * (-gradient(xk)))


def function_alpha2(alpha, xk, Hk):
    return function(xk - alpha * np.dot(Hk, gradient(xk)))


def fast_gradient_method(f, fprime, x0, maxiter=10000, epsi=10e-3):
    df = pd.DataFrame()

    iter_counter: int = 0
    xk = x0
    current_gradient = fprime(x0)
    alpha = 0.005
    while ln.norm(current_gradient) > epsi:
        a, b = search_minimal_segment(alpha, epsi, function_alpha, xk)
        alpha = fibonacci(a, b, epsi, function_alpha, xk)
        xNext = xk + alpha * current_gradient
        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': alpha, 'DX': xNext[0] - xk[0],
                         'DY': xNext[1] - xk[1], 'DF': f(xNext) - f(xk), 'Gradient': current_gradient}]
        df = df.append(current_iter, ignore_index=True)
        xk = xNext
        current_gradient = gradient(xk)
        iter_counter += 1
    df.to_excel('Report.xlsx')
    return xk



def function(x):
    return -3 / (1 + (x[0] - 2)**2 + (x[1] - 2)**2 / 4) - 2 / (1 + (x[0] - 2)**2 / 9 + (x[1] - 3)**2)



def gradient(x):
    grad = nd.Gradient(function)
    dx, dy = grad([x[0], x[1]])
    return np.array([dx, dy])


def second_Pearson(f, fprime, x0, maxiter=None, epsi=10e-3):
    if maxiter is None:
        maxiter = len(x0) * 200

    df = pd.DataFrame()

    k = 0
    current_gradient = fprime(x0)
    N = len(x0)

    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    alpha_k = 0.005

    while ln.norm(current_gradient) > epsi and k < maxiter:
        pk = np.dot(Hk, current_gradient)
        a, b = search_minimal_segment2(alpha_k, epsi, function_alpha2, xk, Hk)
        alpha_k = fibonacci2(a, b, epsi, function_alpha2, xk, Hk)

        xkp1 = xk - alpha_k * pk

        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': alpha_k, 'DX': xkp1[0] - xk[0],
                         'DY': xkp1[1] - xk[1], 'DF': f(xkp1) - f(xk), 'Gradient': current_gradient, 'Hk': Hk}]
        df = df.append(current_iter, ignore_index=True)

        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - current_gradient
        current_gradient = gfkp1
        k += 1

        temp = 1 / np.dot(yk[np.newaxis, :], sk)
        Hk = Hk + ((np.dot(sk - np.dot(Hk, yk), sk[:np.newaxis])) * temp)

    df.to_excel("Report_Second_Pearson.xlsx")
    return (xk, k)


def third_Pearson(f, fprime, x0, maxiter=None, epsi=10e-3):
    if maxiter is None:
        maxiter = len(x0) * 200

    df = pd.DataFrame()

    k = 0
    current_gradient = fprime(x0)
    N = len(x0)

    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    alpha_k = 0.005

    while ln.norm(current_gradient) > epsi and k < maxiter:
        pk = np.dot(Hk, current_gradient)
        a, b = search_minimal_segment2(alpha_k, epsi, function_alpha2, xk, Hk)
        alpha_k = fibonacci2(a, b, epsi, function_alpha2, xk, Hk)

        xkp1 = xk - alpha_k * pk

        current_iter = [{'X': xk[0], 'Y': xk[1], 'F': f(xk), 'Lambda': alpha_k, 'DX': xkp1[0] - xk[0],
                         'DY': xkp1[1] - xk[1], 'DF': f(xkp1) - f(xk), 'Gradient': current_gradient, 'Hk': Hk}]
        df = df.append(current_iter, ignore_index=True)

        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - current_gradient
        current_gradient = gfkp1
        k += 1

        temp = 1 / np.dot(np.dot(yk[np.newaxis, :], Hk), yk)
        A = Hk + ((np.dot(sk - np.dot(Hk, yk), np.dot(yk, Hk).transpose())) * temp)
        Hk = A

    df.to_excel("Report_Third_Pearson.xlsx")
    return (xk, k)


xk = fast_gradient_method(function, gradient, np.array([1, 1]))
result, k = third_Pearson(function, gradient, np.array([1, 1]))
result1, k2 = second_Pearson(function, gradient, np.array([1, 1]))