import numpy as np
from mcholmz import modifiedChol


def newtons_method(x: np.ndarray, function=None):
    if function is None:
        print("Error - no function was given")
        return
    solution_path = [x]  # will contain the points in the solution's path
    alpha = 1
    stop_criteria = 10 ** -5
    gradient = function.gradient(x)
    while np.linalg.norm(gradient) >= stop_criteria:
        direction = get_newton_direction(function.hessian(x), -gradient)
        alpha = get_inexact_step_size(function, alpha, x, direction)

        x = x + alpha * direction
        solution_path.append(x)
        gradient = function.gradient(x)
    return solution_path[-1]


def get_newton_direction(A, b):  # perform LDL decomposition on Ax = b system

    # setting A matrix
    L, d, _ = modifiedChol(A)
    D = np.diag(d.flatten())

    # setting Ly=b
    y = np.zeros((b.size, 1))

    # forward substitution
    for m in range(y.size):
        temp_sum = sum(L[m, i] * y[i, 0] for i in range(m))
        y[m, 0] = (b[m, 0] - temp_sum) / L[m, m]

    B = D @ L.T  # now we need to do backward substitution
    d = np.zeros((y.size, 1))

    for m in range(y.size - 1, -1, -1):
        temp_sum = sum(B[m, i] * d[i, 0] for i in range(y.size - 1, m, -1))
        d[m, 0] = (y[m, 0] - temp_sum) / B[m, m]

    return d


def get_inexact_step_size(f, alpha: float, x: np.ndarray,
                          direction: np.ndarray) -> float:
    beta = 0.5
    sigma = 0.25
    c = f.gradient(x).T @ direction
    threshold = sigma * c
    while threshold * alpha <= f(x + alpha * direction) - f(x):
        alpha *= beta
    return alpha
