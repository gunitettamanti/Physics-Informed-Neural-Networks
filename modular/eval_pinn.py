""" PINN implementation of Porous Media Equation"""
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import PME
import numpy as np
import time as time
import matplotlib.pyplot as plt
import json

from mod import generate_domain, gaussian

with open('params.json') as f:
    params = json.load(f)

layers  = params['layers']
ld = params['ld']

Nx = params['Nx']
Nt = params['Nt']
x0 = params['x0']
x1 = params['x1']
t0 = params['t0']
t1 = params['t1']

(input_data, 
 output_data,
 lambda_data,
 lambda_phys,
 xx,) = generate_domain(x0, x1, t0, t1, Nx, Nt, ld)

PINN = PhysicsInformedNN(layers,
                         dest='./', 
                         activation='tanh',
                         norm_in=[np.array([t0, x0]), np.array([t1, x1])],
                         optimizer=keras.optimizers.Adam(1e-4),
                         restore=True)

# initial condition
points = np.array([(0.0, xi) for xi in xx]).astype('float32')
out = PINN.model(points)[0][:, 0]
plt.figure()
plt.plot(xx, gaussian(xx, 0.5), label='Initial data')
plt.plot(xx, out, label='Initial PINN')

# end condition
# points = np.array([(t1, xi) for xi in xx]).astype('float32')
# out = PINN.model(points)[0]
# plt.plot(xx, out, label='Final PINN')
# plt.legend()

# Residuals
points = np.array([(t1/2, xi) for xi in xx]).astype('float32')
u_t, u_xx = PME(PINN.model, points, [], separate_terms=True)
plt.figure()
plt.plot(u_t)
plt.plot(u_xx)
out2 = out**2
dout = np.gradient(out2, 4.0/Nx)
d2out = np.gradient(dout, 4.0/Nx)
plt.plot(d2out)

# loss
ep, ld, lp = np.loadtxt('output.dat', unpack=True)
plt.figure()
plt.semilogy(ep, ld)
plt.semilogy(ep, lp)

# show
plt.show()
