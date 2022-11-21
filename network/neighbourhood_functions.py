from math import exp, atanh
import numpy as np


def euclidean(n, c):
    return np.linalg.norm(c-n)


def hyperbolic_distance(n, c):
    return 2 * atanh(abs((c-n)/(1-c*np.conj(n))))


class Gaussian:
    """
    Gaussian neighbourhood function
    """
    def __init__(self, step_size=1000, distance_measurement_function=euclidean, sigma_initial=10, sigma_final=1):
        self.sigma = sigma_initial
        self.sigma_final = sigma_final
        self.step = (self.sigma_final-self.sigma) / step_size
        self.dist = distance_measurement_function

    def __call__(self, n, c):
        res = exp(-(self.dist(n, c)**2 / (2*(self.sigma**2))))
        self.decrease_sigma()
        return res

    def decrease_sigma(self):
        if self.sigma + self.step <= self.sigma_final:
            self.sigma = self.sigma_final
        else:
            self.sigma += self.step

class WinnerTakesAll:
    def __init__(self):
        pass

    def __call__(self, n, c):
        if n == c:
            return 1
        return 0
