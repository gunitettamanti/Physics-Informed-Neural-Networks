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

for lr in params['lrs']:
    tot_eps = params['tot_eps']
    PINN.optimizer.learning_rate.assign(lr)
    PINN.train(input_data, 
               output_data,
               PME,
               flags=flags,
               epochs=tot_eps,
               batch_size=params['batch_size'],
               lambda_data=lambda_data,
               lambda_phys=lambda_phys,
               )
