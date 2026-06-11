import math
import os
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "xdg-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import curve_fit
from scipy.special import eval_legendre


DATA_DIR = Path(__file__).resolve().parent

def read_dat_file(file_path):
    data = np.genfromtxt(DATA_DIR / file_path, names=True)
    return np.column_stack([data[name] for name in data.dtype.names])

def write_dat_file(file_path, data):
    np.savetxt(file_path, data)

# getting the angles, interaction potentials
# converting angles to radians
data = read_dat_file('R_7p3_avqzmb.dat')
thetas, potcis = np.radians(data[:, 1]), data[:, 2]


def print_data(thetas, potcis):
    for i in range(len(thetas)):
        print(f"Theta: {thetas[i]}, Potential: {potcis[i]}")


def legendre_fit(thetas, potcis, max_degree):
    degrees = list(range(0, max_degree + 1, 2))
    x = np.cos(thetas)

    A = np.column_stack([eval_legendre(n, x) for n in degrees])
    coeffs, *_ = np.linalg.lstsq(A, potcis, rcond=None)

    return degrees, coeffs


def legendre_fit_func(theta, degrees, coeffs):
    x = np.cos(theta)
    return sum(c * eval_legendre(n, x) for n, c in zip(degrees, coeffs))


def plot_point_and_fit(thetas, potcis, fit_func):
    plt.scatter(thetas, potcis, label="Data Points")
    theta_fit = np.linspace(min(thetas), max(thetas), 100)
    plt.plot(theta_fit, fit_func(theta_fit), color="red", label="Fit")
    plt.xlabel("Theta (radians)")
    plt.ylabel("Potential")
    plt.legend()
    plt.show()

# Perform the fit and return the error
def fit_and_plot(thetas, potcis, degree):
    coeffs = legendre_fit(thetas, potcis, degree)
    fit_func = lambda theta: legendre_fit_func(theta, *coeffs)
    print("_______data________")
    print(fit_func)
    print(coeffs)
    plot_point_and_fit(thetas, potcis, fit_func)
    return np.sum((potcis - fit_func(thetas)) ** 2)

# Example usage
if __name__ == "__main__":
    degree = 50

    # fit the model, state function, state error, and plot results
    error = fit_and_plot(thetas, potcis, degree)
    print(f"Fitting error (degree {degree}): {error}")