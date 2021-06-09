import random
from math import sqrt
import numpy as np


def golden_ratio(function, a, b, eps):
    SQRT5 = sqrt(5)

    x1 = a + (3 - SQRT5) / 2 * (b - a)
    x2 = a + (SQRT5 - 1) / 2 * (b - a)

    f1 = function(x1)
    f2 = function(x2)
    i: int = 0

    while abs(a - b) > eps:
        if f1 < f2:
            b = x2
            x2 = x1
            x1 = a + (3 - SQRT5) / 2 * (b - a)
            f2 = f1
            f1 = function(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (SQRT5 - 1) / 2 * (b - a)
            f1 = f2
            f2 = function(x2)

        i += 1
    return (a + b) / 2, i


def search_minimal_segment(function, x0, eps):
    number_of_calc = 0
    h: float = 0
    first_x: float = x0
    current_x: float = 0
    next_x: float = 0
    first_f: float = function(first_x)

    if first_f > function(first_x + eps):
        current_x = first_x + eps
        h = eps
    elif first_f > function(first_x - eps):
        current_x = first_x - eps
        h = - eps

    h = 2 * h
    current_f = function(current_x)
    next_f = function(current_x + h)
    while current_f > next_f:
        h = 2 * h
        number_of_calc += 1

        next_x = current_x + h
        first_x = current_x
        current_x = next_x

        current_f = next_f
        next_f = function(next_x)

    return first_x, next_x


def calculate_vector_sum(function, xk, m, g, current_value, field_x, field_y):
    res = [0, 0]

    for i in range(m):
        eps = calculate_x(xk, field_x, field_y)
        res += eps * (function(xk + g * eps) - current_value)

    return res


def calculate_x():
    x = random.uniform(-1, 1)
    y = sqrt(1 - x**2)
    return np.array([x, y])
