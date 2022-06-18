import numpy as np
from newtons_method import newtons_method
from matplotlib import pyplot as plt


class PenaltyFunction:
    def __call__(self, x: float) -> float:
        if x >= -0.5:
            return 0.5 * x ** 2 + x
        return -0.25 * (np.log10(-2 * x)) - 3 / 8

    def gradient(self, x: float) -> float:
        if x >= -0.5:
            return float(x + 1)
        return -1 / (4 * x * (np.log(10)))

    def hessian(self, x: float) -> float:
        if x >= -0.5:
            return 1
        return 1 / (4 * np.log(10) * x ** 2)


class F:
    def __init__(self, function, constrain_list: list, phi, p):
        self.function = function
        self.constrains = constrain_list
        self.lambdas = [np.zeros((len(self.constrains), 1))]
        self.penalty = phi
        self.p = p

    def __call__(self, x):
        res = self.function(x)
        for g in self.constrains:
            res += (1 / self.p) * self.penalty(self.p * g(x))
        return res

    def gradient(self, x):
        res = self.function.gradient(x)
        # for _lambda, g in zip(self.lambdas, self.constrains):
        #     res += _lambda * g.gradient(x)
        for i in range(len(self.constrains)):
            res += self.penalty.gradient(self.p * self.constrains[i](x)) * self.constrains[i].gradient(x)
        return res

    def hessian(self, x):
        res = self.function.hessian(x)
        for g in self.constrains:
            a = self.p * self.penalty.hessian(self.p * g(x)) * (g.gradient(x) @ g.gradient(x).T)
            b = self.penalty.gradient(self.p * g(x)) * g.hessian(x)
            res += a + b
        return res

    def get_lambdas(self):
        return self.lambdas

    def update_lambdas(self, x):
        new_lambdas = np.zeros((len(self.constrains), 1))
        for i in range(len(self.constrains)):
            new_lambdas[i, 0] = self.penalty.gradient(self.p * self.constrains[i](x))
        self.lambdas.append(new_lambdas)
        return

    def set_p(self, new_p):
        self.p = new_p
        return


class f:
    def __call__(self, x: np.ndarray) -> float:
        return 2 * (x[0, 0] - 5) ** 2 + (x[1, 0] - 1) ** 2

    def gradient(self, x: np.ndarray) -> np.ndarray:
        dx1 = 4 * (x[0, 0] - 5)
        dx2 = 2 * (x[1, 0] - 1)
        res = np.array([[float(dx1)], [float(dx2)]])
        return res

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[4.0, 0.0], [0.0, 2.0]])


class g1:
    def __call__(self, x: np.ndarray) -> float:
        return 0.5 * x[0, 0] + x[1, 0] - 1

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([[0.5], [1.0]])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[0.0, 0.0], [0.0, 0.0]])


class g2:
    def __call__(self, x: np.ndarray) -> float:
        return x[0, 0] - x[1, 0]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1.0], [-1.0]])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[0.0, 0.0], [0.0, 0.0]])


class g3:
    def __call__(self, x: np.ndarray) -> float:
        return -x[0, 0] - x[1, 0]

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([[-1.0], [-1.0]])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[0.0, 0.0], [0.0, 0.0]])


def penalty_aggregate(function, constrains_list: list, phi):
    p = 0.1
    x_path = [np.array([[0], [0]])]
    alpha = 2
    iterations = 0
    augmented_gradient = []  # values of Augmented Lagrangian Gradients
    max_violations = []
    f_vals_diff = []
    x_vals_diff = []
    x_vals = []
    lambdas_diff = []
    x_opt = np.array([[2 / 3], [2 / 3]])
    f_opt = function(x_opt)
    lambda_opt = np.array([[12], [34 / 3], [0]])
    F1 = F(function, constrains_list, phi, p)
    F1.update_lambdas(x_path[-1])
    while p < 10 ** 5:
        x_path = newtons_method(x_path[-1], F1)
        F1.update_lambdas(x_path[-1])

        # Update graphs data
        for x in x_path:
            augmented_gradient.append(np.linalg.norm(F1.gradient(x)))
            max_violation = 0
            for g in constrains_list:
                max_violation = max(max_violation, g(x))
            max_violations.append(max_violation)
            f_vals_diff.append(abs(function(x) - f_opt))
            x_vals_diff.append(np.linalg.norm(x - x_opt))
            x_vals.append(x)
            lambdas = F1.get_lambdas()
            lambdas_diff.append(np.linalg.norm(lambdas[-1] - lambda_opt))

        F1.set_p(p)
        p = alpha * p
        iterations += len(x_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(list(range(iterations)), augmented_gradient)
    ax1.set(yscale="log", title="Augmented Lagrangian Gradients")
    ax1.grid()
    ax2.plot(list(range(iterations)), f_vals_diff)
    ax2.set(yscale="log", title=r'|$f(x_k)$ - $f(x^*)$|')
    ax2.grid()
    ax3.plot(list(range(iterations)), max_violations)
    ax3.set(yscale="log", title="Maximal Constraint Violation")
    ax3.grid()
    ax4.plot(list(range(iterations)), x_vals_diff, label=r'||$x_k$ - $x^*$||')
    ax4.plot(list(range(iterations)), lambdas_diff, label=r'||$\lambda_k$ - $\lambda^*$||')
    ax4.set(yscale="log", title=r'||$x_k$ - $x^*$|| & ||$\lambda_k$ - $\lambda^*$||')
    ax4.grid()
    ax4.legend()
    plt.show()
    print(f"x = \n{x_vals[-1]}\nlambda = \n{lambdas[-1]}")
    return x_path[-1], lambdas
