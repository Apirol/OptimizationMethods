from math import sqrt


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
    return (a + b) / 2


def golden_ratio_2(function, a, b, eps):
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
    return (a + b) / 2


def search_minimal_segment(x0, eps, function, xk):
    number_of_calc = 0
    h: float = 0
    first_x: float = x0
    current_x: float = 0
    next_x: float = 0
    first_f: float = function(first_x, xk)

    if first_f > function(first_x + eps, xk):
        current_x = first_x + eps
        h = eps
    elif first_f > function(first_x - eps, xk):
        current_x = first_x - eps
        h = - eps
    else:
        return first_x + eps, first_x - eps

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
    else:
        return first_x + eps, first_x - eps

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
