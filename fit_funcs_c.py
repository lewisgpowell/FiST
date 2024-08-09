# Python interface to the C++ numerical functions using ctypes along with code for the nonlinear fitting routine
# Copyright (C) 2024 Lewis Powell
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/. 

import ctypes as ct
import numpy as np
import numpy.ctypeslib as cl
import os
#import sys
from pathlib import Path
import nlopt

# Load binaries compiled from C++ code
c_dir = Path(__file__).parent
current_dir = os.getcwd()
os.chdir(c_dir / 'bin')
c_lib = cl.load_library("Fitting_Funcs", ".")
os.chdir(current_dir)

# define C data type for a pointer to double
array = cl.ndpointer(dtype=np.dtype(float))

# Dyne function
c_lib.dyne_func.restype = ct.c_double
def dyne_func_uv(E, N, D, L):
    return c_lib.dyne_func(ct.c_double(E), ct.c_double(N), ct.c_double(D), ct.c_double(L))
dyne_func = np.vectorize(dyne_func_uv)

# Fermi-Dirac derivative
c_lib.fermi_dirac_deriv.restype = ct.c_double
def fermi_dirac_deriv_uv(E, T):
    return c_lib.fermi_dirac_deriv(ct.c_double(E), ct.c_double(T))
fermi_dirac_deriv = np.vectorize(fermi_dirac_deriv_uv)

# Dyne fit
c_lib.dyne_fit.restype = None # return type is void
c_lib.dyne_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def dyne_fit(V, N, D, L, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.dyne_fit(G, len(G), N, D, L, T)
    return G

c_lib.dyne_curve_fit.restype = None
c_lib.dyne_curve_fit.argtypes = [array, array, ct.c_double, ct.c_size_t, array, array]
def dyne_curve_fit(voltages, conductances, temperature, initial_guess):
    final_params = np.zeros(3)
    c_lib.dyne_curve_fit(voltages, conductances, temperature, len(voltages), initial_guess, final_params)
    return final_params

c_lib.btk_curve_fit.restype = None
c_lib.btk_curve_fit.argtypes = [array, array, ct.c_double, ct.c_double, ct.c_size_t, array, array]
def btk_curve_fit(voltages, conductances, barrier, temperature, initial_guess):
    final_params = np.zeros(3)
    c_lib.btk_curve_fit(voltages, conductances, temperature, barrier, len(voltages), initial_guess, final_params)
    return final_params

# Dyne skew fit
c_lib.dyne_skew_fit.restype = None
c_lib.dyne_skew_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def dyne_fit_skew(V, Na, Nb, D, L, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.dyne_skew_fit(G, len(G), Na, Nb, D, L, T)
    return G

# Gap temperature dependence
c_lib.gap_at_temp.restype = ct.c_double
def gap_at_temp_uv(T, D0):
    return c_lib.gap_at_temp(ct.c_double(T), ct.c_double(D0))
gap_at_temp = np.vectorize(gap_at_temp_uv)

# Point node density of states
c_lib.dos_point_node.restype = ct.c_double
def dos_point_node_uv(E, N, D, L):
    return c_lib.dos_point_node(ct.c_double(E), ct.c_double(N), ct.c_double(D), ct.c_double(L))
dos_point_node = np.vectorize(dos_point_node_uv)

# Point node fit
c_lib.point_node_fit.restype = None
c_lib.point_node_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def point_node_fit(V, N, D, L, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.point_node_fit(G, len(G), N, D, L, T)
    return G

# Line node fit
c_lib.line_node_fit.restype = None
c_lib.line_node_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def line_node_fit(V, N, D, L, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.line_node_fit(G, len(G), N, D, L, T)
    return G

# Line node density of states
c_lib.dos_line_node.restype = ct.c_double
def dos_line_node_uv(E, N, D, L):
    return c_lib.dos_line_node(ct.c_double(E), ct.c_double(N), ct.c_double(D), ct.c_double(L))
dos_line_node = np.vectorize(dos_line_node_uv)

# Combined (nodeless and line node) fit
c_lib.combined_fit.restype = None
c_lib.combined_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def combined_fit(V, N, Ds, Dp, L, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.combined_fit(G, len(G), N, Ds, Dp, L, T)
    return G

# BTK fit
c_lib.btk_fit.restype = None
c_lib.btk_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def btk_fit(V, N, D, L, Z, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.btk_fit(G, len(G), N, D, L, Z, T)
    return G

# Get roots of 'Maki equation'
c_lib.maki_equation_ctype.restype = None
c_lib.maki_equation_ctype.argtypes = [array, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_bool]
def maki_roots(E, D, B, Z, is_spin_up):
    results = np.zeros(2, dtype=np.float64)
    c_lib.maki_equation_ctype(results, E, D, B, Z, is_spin_up)
    return results

# Density of states for fully gapped superconductor with Maki broadening
c_lib.maki_dos.restype = ct.c_double
def maki_dos_uv(E, N, D, Z, B):
    return c_lib.maki_dos(ct.c_double(E), ct.c_double(N), ct.c_double(D), ct.c_double(Z), ct.c_double(B))
maki_dos = np.vectorize(maki_dos_uv)

# Fit for Dyne's model with Maki orbital angular momentum broadening parameter
c_lib.maki_fit.restype = None
c_lib.maki_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def maki_fit(V, N, D, Z, B, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.maki_fit(G, len(G), N, D, Z, B, T)
    return G

# BTK fit with Maki broadening
c_lib.btk_maki_fit.restype = None
c_lib.btk_maki_fit.argtypes = [array, ct.c_size_t, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double]
def btk_maki_fit(V, N, D, xi, Z, B, T):
    G = V.copy() # Deep-copy array so that x-data does not get overwritten
    c_lib.btk_maki_fit(G, len(G), N, D, xi, Z, B, T)
    return G

# Functions for estimating errors on the fitted parameters
def finite_difference(function, xdata, fit_params, external_params, param_i, h=1e-3):
    varying_param_up = fit_params[param_i] + h
    varying_param_down = fit_params[param_i] - h
    params_upper = np.concatenate([fit_params[0:param_i], [varying_param_up], fit_params[param_i:-1]])
    params_lower = np.concatenate([fit_params[0:param_i], [varying_param_down], fit_params[param_i:-1]])
    deriv = ( function(xdata, *params_upper, *external_params) - function(xdata, *params_lower, *external_params) ) / (2*h)
    return deriv

def estimate_jacobian(function, xdata, fit_params, external_params, h=1e-3):
    jacobian = np.zeros(shape=(len(fit_params),len(xdata)))
    for param_i, param in enumerate(fit_params):
        jacobian[param_i] = finite_difference(function, xdata, fit_params, external_params, param_i, h)
    return jacobian

# Return sum of square residuals for a model fit_function, fiting parameters, external parameters and x, y data
def sum_of_square_residuals(fit_function, xdata, ydata, fit_params, external_params):
    residuals = fit_function(xdata, *fit_params, *external_params) - ydata
    sum_of_squares = np.sum(np.square(residuals))
    return sum_of_squares

# Return Pseudo-Huber loss
def huber_loss(fit_function, xdata, ydata, fit_params, external_params, scale):
    residuals = fit_function(xdata, *fit_params, *external_params) - ydata
    huber_func = lambda x, s: s**2 * (np.sqrt(1 + (x / s)**2) - 1)
    sum_of_loss = np.sum(huber_func(residuals, scale))
    return sum_of_loss

# Implement optimisation
def curve_fit(model_function, xdata, ydata, initial_guess, external_params=np.array([]), lower_bounds=-np.inf, upper_bounds=np.inf, h=1e-3, loss='squares', scale=0.1, get_cov=True):
    if loss=='squares':
        minimisation_function = lambda fit_x, grad: sum_of_square_residuals(model_function, xdata, ydata, fit_x, external_params)
    elif loss=='huber':
        minimisation_function = lambda fit_x, grad: huber_loss(model_function, xdata, ydata, fit_x, external_params, scale)
    else:
        print("Undefined loss function")
    opt = nlopt.opt(nlopt.LN_COBYLA, len(initial_guess))
    opt.set_min_objective(minimisation_function)
    opt.set_xtol_rel(1e-6)
    opt.set_ftol_rel(1e-6)
    opt.set_maxeval(1000)
    opt.set_lower_bounds(lower_bounds)
    opt.set_upper_bounds(upper_bounds)
    param_opt = np.array(opt.optimize(np.array(initial_guess,dtype=np.dtype(np.float64))))
    
    # Get covariance matrix estimate
    if get_cov:
        jacobian = np.matrix(estimate_jacobian(model_function, xdata, param_opt, external_params, h))
        mean_variance = opt.last_optimum_value()/(len(xdata)-len(initial_guess))
        reduced_jacobian = np.matmul(jacobian, jacobian.transpose())
        if (np.linalg.det(reduced_jacobian) == 0):
            print("Warning: Singular Jacobian, errors undefined")
            covariance = np.zeros_like(reduced_jacobian)
        else:
            covariance = np.linalg.inv(reduced_jacobian) * mean_variance
    #covariance = np.identity(len(param_opt))
        return (param_opt, covariance)
    return param_opt

## Function to get a single spectrum, optionally only within a certain voltage range
def get_spectrum(data, B, cutoff=None):
    dataset = data[data["B"] == B]
    dataset = dataset.sort_values("V")
    if cutoff:
        x = dataset[abs(dataset["V"]) < cutoff]["V"].to_numpy()
        y = dataset[abs(dataset["V"]) < cutoff]["G"].to_numpy()
    else:
        x = dataset["V"].to_numpy()
        y = dataset["G"].to_numpy()
    return (x, y)
