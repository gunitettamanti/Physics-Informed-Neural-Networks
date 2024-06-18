""" PINN implementation of Porous Media Equation"""
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

import numpy as np

def gaussian(x ,s):
    return (1.0/np.sqrt(2.0*np.pi*s**2))*np.exp(-x**2/(2.0*s**2))

def generate_domain(x0, x1, t0, t1, Nx, Nt, lp):
    # Generate domain
    xx = np.linspace(x0, x1, Nx)
    tt = np.linspace(t0, t1, Nt)

    lambda_data = []
    lambda_phys = []
    input_data  = []
    output_data = []
    flags       = []
    ld = 1.0

    # Initial conditions
    input_data  += [[0.0, xi] for xi in xx]
    output_data += [gaussian(xi, 0.5) for xi in xx]
    lambda_data += [ld]*Nx
    lambda_phys += [lp]*Nx
    flags       += [1]*Nx

    # Boundary conditions
    # input_data  += [(0.0, -2.0)]*Nt
    # output_data += [0.0]*Nt
    # lambda_data += [ld]*Nt
    # lambda_phys += [0.0]*Nt
    #   
    # input_data  += [(0.0,  2.0)]*Nt
    # output_data += [0.0]*Nt
    # lambda_data += [ld]*Nt
    # lambda_phys += [0.0]*Nt

    # Collocation points
    for ti in tt[1:]:
        input_data  += [[ti, xi] for xi in xx]
        output_data += [0.0]*Nx
        lambda_data += [0.0]*Nx
        lambda_phys += [lp]*Nx
        flags       += [0]*Nx

    input_data  = np.array(input_data)
    output_data = np.array(output_data).reshape(-1, 1)
    lambda_data = np.array(lambda_data)
    lambda_phys = np.array(lambda_phys)
    flags       = np.array(flags)

    input_data = input_data.astype(np.float32)
    output_data = output_data.astype(np.float32)
    lambda_data = lambda_data.astype(np.float32)
    lambda_phys = lambda_phys.astype(np.float32)
    flags       = flags.astype(np.float32)

    return input_data, output_data, lambda_data, lambda_phys, flags, xx
