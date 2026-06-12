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
def fit_and_plot(thetas, potcis, degree, plot=False):
    coeffs = legendre_fit(thetas, potcis, degree)
    fit_func = lambda theta: legendre_fit_func(theta, *coeffs)
    print("_______data________")
    print(fit_func)
    print(coeffs)
    if plot:
        plot_point_and_fit(thetas, potcis, fit_func)
    return np.sum((potcis - fit_func(thetas)) ** 2)

# Use corrected Akaike information criterion to decide the number of terms
def corrected_aic(n, error, k):
    return n * np.log(error / n) + 2 * k

def select_degree(thetas, potcis, max_degree):
    best_aic = np.inf
    best_degree = 0

    for degree in range(2, max_degree + 1, 2):
        error = fit_and_plot(thetas, potcis, degree, plot=False)
        aic = corrected_aic(len(thetas), error, degree)
        print(f"AIC (degree {degree}): {aic}")

        if aic < best_aic:
            best_aic = aic
            best_degree = degree

    print(f"Best degree: {best_degree}")
    return best_degree


# Find optimal degree by bounding the error
def smallest_degree_below_error(thetas, potcis, max_degree, error_threshold):
    for degree in range(2, max_degree + 1, 2):
        error = fit_and_plot(thetas, potcis, degree, plot=False)
        if error < error_threshold:
            print(f"Smallest degree below error threshold {error_threshold}: {degree}")
            return degree
    print(f"No degree found below error threshold {error_threshold}")
    return None

# find optimal degree by determining percentage change in RMSE
# when adding the new term: if the change is less than a certain percentage, we stop.
def percentage_change_threshold(thetas, potcis, max_degree, percentage_threshold):
    previous_error = float('inf')
    previous_degree = None

    for degree in range(2, max_degree + 1, 2):
        error = fit_and_plot(thetas, potcis, degree, plot=False)
        if previous_error != float('inf'):
            percentage_change = (previous_error - error) / previous_error * 100
            print(f"Percentage change (degree {degree}): {percentage_change:.2f}%")
            if percentage_change < percentage_threshold:
                print(f"Stopping at degree {degree} due to percentage change threshold {percentage_threshold}%")
                return previous_degree
        previous_error = error
        previous_degree = degree

    print(f"Reached max degree {max_degree} without meeting percentage change threshold {percentage_threshold}%")
    return previous_degree

# Example usage
if __name__ == "__main__":
    degree = 8

    # fit the model, state function, state error, and plot results
    error = fit_and_plot(thetas, potcis, degree, plot=True)
    print(f"Fitting error (degree {degree}): {error}")

    # optimal degree by AIC
    best_degree = select_degree(thetas, potcis, 50)

    # optimal degree by bounding error
    smallest_degree = smallest_degree_below_error(thetas, potcis, 50, error_threshold=2e-1)

    # optimal degree by percentage change in RMSE
    percentage_degree = percentage_change_threshold(thetas, potcis, 50, percentage_threshold=1)