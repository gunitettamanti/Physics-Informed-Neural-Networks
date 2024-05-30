""" PINN implementation of Porous Media Equation"""
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import PME
import numpy as np
import time as time
import matplotlib.pyplot as plt

lr = 1e-4
layers  = [2] + 3*[64] + [1]

PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='tanh', #probar con tanh 
                         optimizer=keras.optimizers.Adam(lr),
                         #optimizer='lbfgs',
                         restore=True)
PINN.model.summary()

def cte_validation(self,X,u):    
    # Definimos una función que el código después llama
    # El único parametro que le pasa código es el número de epoch
    # El resto lo definimos al generar la función
    def validation(ep):        
        # Get prediction                                
        Y  = self.model(X)[0].numpy()

        u_p = Y[:,0]       

        # True value
        sol  = u(X)
        
        # Error global
        err  = np.sqrt(np.mean((u_p-sol)**2))/np.std(sol)
        
        # Loss functions
        output_file = open(self.dest + 'validation.dat', 'a')
        print(ep,err,file=output_file)
        output_file.close()       
        
        dx = 4/len(X)
        fx = tf.cumsum(u_p)
        fx = fx*dx     
        Ix = tf.ones_like(u_p)*(fx[-1])           

        output_file_1 = open(self.dest + 'mass_conservation.dat', 'a')
        print(ep,f'{Ix[-1].numpy()}',file=output_file_1)
        output_file_1.close()    
        
    return validation
def gaussian( x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

Nr = 20000
Ncb = 100
Nd = 100

tmin = 0
tmax = 0.05
xmin = -2
xmax = 2

lb = np.array([tmin,xmin])
ub = np.array([tmax,xmax])

#Cond ini
t_0 = np.ones([Nd,1])*lb[0]
x_0 = np.random.uniform(xmin,xmax,(Nd,1))
X_0 = np.concatenate([t_0,x_0],axis=1)

#Cond borde
t_cb = np.random.uniform(tmin,tmax,(Ncb,1))
x_cb = lb[1] + (ub[1] - lb[1])*np.random.binomial(1,0.5,(Ncb,1))
X_cb = np.concatenate([t_cb,x_cb],axis=1)

#Colllocations points
t_r = np.random.uniform(tmin,tmax,(Nr,1))
x_r = np.random.uniform(xmin,xmax,(Nr,1))
X_r = np.concatenate([t_r,x_r],axis=1)

X = np.concatenate([X_0,np.concatenate([X_r,X_cb],axis=0)],axis=0)

#Condicion inicial gaussiana con una dispersion de 0.5
Y = gaussian(X_0[:,1],0.5).reshape(Nd,1)

#cond_ini = np.loadtxt('0.34.py', delimiter=',')
#Y[:Nx] = cond_ini[:,0].reshape(len(cond_ini),1)

tot_point = Ncb+Nd+Nr
lambda_data = np.zeros(tot_point)
lambda_phys = np.zeros(tot_point)
lambda_bc = np.zeros(tot_point)

lambda_phys[Nd:Nd+Nr] = 1
lambda_bc[-Ncb:] = 1

tot_eps = 10000

PINN.validation = cte_validation(PINN,X,Y)

t1 = time.time()
exp = 6
for i in range(np.arange(1,7)):
    lambda_data[:Nd] = 1*10**exp
    PINN.train(X, Y, PME,
           epochs=tot_eps,
           batch_size=tot_point,
           #eq_params=eq_params,           
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,
           lambda_bc=lambda_bc, 
           #flags=flags,               # Separa el dataset a cada t
           #rnd_order_training=False,  # No arma batches al hacer
           #alpha=alpha,
           verbose=False,            
           valid_freq=1000,
           timer=False,
           data_mask=[True,False])
t2 = time.time()
print(int(t2-t1)/60)


