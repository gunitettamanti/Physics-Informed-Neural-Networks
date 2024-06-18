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
lp = params['lp']

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
 flags,
 xx,) = generate_domain(x0, x1, t0, t1, Nx, Nt, lp)

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

points = np.array([(t1/2, xi) for xi in xx]).astype('float32')
out = PINN.model(points)[0][:, 0]
plt.plot(xx, out, label='PINN t1/2')

# x=0
tt = np.linspace(t0, t1, Nt)
points = np.array([(ti, 0.0) for ti in tt]).astype('float32')
out = PINN.model(points)[0][:, 0]
plt.figure()
plt.plot(tt, out)

# end condition
# points = np.array([(t1, xi) for xi in xx]).astype('float32')
# out = PINN.model(points)[0]
# plt.plot(xx, out, label='Final PINN')
# plt.legend()

# Residuals
points = np.array([(0.0, xi) for xi in xx]).astype('float32')
out = PINN.model(points)[0][:, 0]
u_t, u_xx = PME(PINN.model, points, [], separate_terms=True)
plt.figure()
plt.plot(u_t, label='u_t')
plt.plot(0.01*u_xx, label='u2_xx')
plt.plot(u_t + 0.01*u_xx, label='residual')
plt.legend()
ff = PME(PINN.model, points, [])
print(np.mean(ff[0]**2), np.mean((u_t+0.01*u_xx)**2))

# loss
ep, ld, lp = np.loadtxt('output.dat', unpack=True)
plt.figure()
plt.semilogy(ep, ld)
plt.semilogy(ep, lp)

# show
plt.show()
