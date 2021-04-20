import numpy as np


def read_massive(filename):
    return np.loadtxt(filename, dtype=np.int)


def fibonacci(a, b, eps, function, xk):
    FIBONACCI = read_massive("fibonacci_numbers.txt")
    n: int = 0
    number_of_calc = 0
    current_length = b - a

    while FIBONACCI[n] < current_length / eps:
        n += 1

    n -= 3
    x1 = a + FIBONACCI[n] / FIBONACCI[n + 2] * (b - a)
    x2 = a + b - x1
    f1 = function(x1, xk)
    f2 = function(x2, xk)
    i: int = 0

    while number_of_calc <= n:
        number_of_calc += 1
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + FIBONACCI[n - number_of_calc + 1] / FIBONACCI[n - number_of_calc + 3] * (b - a)
            f2 = f1
            f1 = function(x1, xk)
        else:
            a = x1
            x1 = x2
            x2 = a + FIBONACCI[n - number_of_calc + 2] / FIBONACCI[n - number_of_calc + 3] * (b - a)
            f1 = f2
            f2 = function(x2, xk)
        i += 1
    return a


def fibonacci2(a, b, eps, function, xk, Hk):
    FIBONACCI = read_massive("fibonacci_numbers.txt")
    n: int = 0
    number_of_calc = 0
    current_length = b - a

    while FIBONACCI[n] < current_length / eps:
        n += 1

    n -= 3
    x1 = a + FIBONACCI[n] / FIBONACCI[n + 2] * (b - a)
    x2 = a + b - x1
    f1 = function(x1, xk, Hk)
    f2 = function(x2, xk, Hk)
    i: int = 0

    while number_of_calc <= n:
        number_of_calc += 1
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + FIBONACCI[n - number_of_calc + 1] / FIBONACCI[n - number_of_calc + 3] * (b - a)
            f2 = f1
            f1 = function(x1, xk, Hk)
        else:
            a = x1
            x1 = x2
            x2 = a + FIBONACCI[n - number_of_calc + 2] / FIBONACCI[n - number_of_calc + 3] * (b - a)
            f1 = f2
            f2 = function(x2, xk, Hk)
        i += 1
    return a


def search_minimal_segment(x0, eps, function, xk):
    number_of_calc = 0
    h: float = 0
    first_x: float = x0
    current_x: float = 0
    next_x: float = 0
    first_f: float = function(first_x, xk)

    if first_f == function(first_x + eps, xk):
        return first_x, first_x + eps
    elif first_f == function(first_x - eps, xk):
        return first_x - eps, first_x

    if first_f > function(first_x + eps, xk):
        current_x = first_x + eps
        h = eps
    elif first_f > function(first_x - eps, xk):
        current_x = first_x - eps
        h = - eps

    h = 2 * h
    current_f = function(current_x, xk)
    next_f = function(current_x + h, xk)
    while current_f > next_f:
        h = 2 * h
        number_of_calc += 1

        next_x = current_x + h
        first_x = current_x
        current_x = next_x

        current_f = next_f
        next_f = function(next_x, xk)

    return first_x, next_x


def search_minimal_segment2(x0, eps, function, xk, Hk):
    number_of_calc = 0
    h: float = 0
    first_x: float = x0
    current_x: float = 0
    next_x: float = 0
    first_f: float = function(first_x, xk, Hk)

    if first_f > function(first_x + eps, xk, Hk):
        current_x = first_x + eps
        h = eps
    elif first_f > function(first_x - eps, xk, Hk):
        current_x = first_x - eps
        h = - eps

    h = 2 * h
    current_f = function(current_x, xk, Hk)
    next_f = function(current_x + h, xk, Hk)
    while current_f > next_f:
        h = 2 * h
        number_of_calc += 1

        next_x = current_x + h
        first_x = current_x
        current_x = next_x

        current_f = next_f
        next_f = function(next_x, xk, Hk)

    return first_x, next_x
