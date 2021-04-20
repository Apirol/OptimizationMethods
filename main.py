import numpy as np
import numpy.linalg as ln
from scipy import optimize
from method import fibonacci,search_minimal_segment
import numdifftools as nd


def function_alpha(alpha, xk):
    return function(xk - alpha * (-gradient(xk)))


def function_alpha2(alpha, Hk, xk):
    return function(xk - alpha * Hk * (-gradient([1, 1])))


def fast_gradient_method(f, fprime, x0, maxiter=10000, epsi=10e-3):
    xk = x0
    current_gradient = fprime(x0)
    alpha = 0.005
    while ln.norm(current_gradient) > epsi:
        a, b = search_minimal_segment(alpha, epsi, function_alpha, xk)
        alpha = fibonacci(a, b, epsi, f, xk)
        xNext = xk + alpha / ln.norm(current_gradient) * current_gradient
        xk = xNext
        current_gradient = gradient(xk)
    print('Искомое x = ' + str(xk))



def function(x):
    return x[0]**2 - x[0]*x[1] + x[1]**2 + 9*x[0] - 6*x[1] + 20



def gradient(x):
    grad = nd.Gradient(function)
    dx, dy = grad([x[0], x[1]])
    return np.array([dx, dy])


def bfgs_method(f, fprime, x0, maxiter=None, epsi=10e-3):
    """
    Minimize a function func using the BFGS algorithm.

    Parameters
    ----------
    func : f(x)
        Function to minimise.
    x0 : ndarray
        Initial guess.
    fprime : fprime(x)
        The gradient of `func`.
    """

    if maxiter is None:
        maxiter = len(x0) * 200

    # initial values
    k = 0
    current_gradient = fprime(x0)
    N = len(x0)
    # Set the Identity matrix I.
    I = np.eye(N, dtype=int)
    Hk = I
    xk = x0
    alpha_k = 0.005

    while ln.norm(current_gradient) > epsi and k < maxiter:

        # pk - direction of search

        pk = np.dot(Hk, current_gradient)

        # Line search constants for the Wolfe conditions.
        # Repeating the line search

        # line_search returns not only alpha
        # but only this value is interesting for us

        a, b = search_minimal_segment(alpha_k, epsi, function_alpha, xk)
        alpha_k = fibonacci(a, b, epsi, function_alpha, xk)

        xkp1 = xk - alpha_k * pk
        sk = xkp1 - xk
        xk = xkp1

        gfkp1 = fprime(xkp1)
        yk = gfkp1 - current_gradient
        current_gradient = gfkp1

        k += 1

        print(yk)
        print(sk[:, np.newaxis])

        temp = 1 / np.dot(yk[np.newaxis, :], sk)
        print(temp)
        A = Hk + ((np.dot(sk - np.dot(Hk, yk), sk[:np.newaxis])) * temp)
        Hk = A

    return (xk, k)


fast_gradient_method(function, gradient, np.array([1, 1]))
result, k = bfgs_method(function, gradient, np.array([1, 1]))

print('Result of BFGS method:')
print('Final Result (best point): %s' % (result))
print('Iteration Count: %s' % (k))