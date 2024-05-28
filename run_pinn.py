""" PINN implementation of Porous Media Equation"""
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import opinion_model 
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


Lx = 4 
Nx = 100
Nt = 100

seq_t = 0.05
t = seq_t*np.random.rand(Nt) 
x = 4*np.random.rand(Nx) - 2

T,X = np.meshgrid(t,x)
 #Ordeno el vector como (t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None]))

#Condicion inicial gaussiana con una dispersion de 0.5
Y = gaussian(X,0.5)

#cond_ini = np.loadtxt('0.34.py', delimiter=',')
#Y[:Nx] = cond_ini[:,0].reshape(len(cond_ini),1)

lambda_data = np.zeros(Nt*Nx) #[1,0,0,..]
lambda_data[:Nx] = 1e6

lambda_phys = np.ones(Nt*Nx)
lambda_phys[:Nx] = 0 #[0,1,1,..]

bc = np.zeros(Nx)
bc[:5] = 1
bc[-5:] = 1
lambda_bc = np.tile(bc,Nt)

n_t_in_batch = Nt #Nt tiene que ser divisble por batches
flags = np.repeat(np.arange(Nt/n_t_in_batch),Nx*n_t_in_batch)

alpha = 0.0
tot_eps = 50000
#eq_params = [n_t_in_batch]

PINN.validation = cte_validation(PINN,X,Y)

t1 = time.time()
PINN.train(X, Y, opinion_model,
           epochs=tot_eps,
           batch_size=n_t_in_batch*Nx,
           #eq_params=eq_params,           
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,
           lambda_bc=lambda_bc, 
           flags=flags,               # Separa el dataset a cada t
           rnd_order_training=False,  # No arma batches al hacer
           alpha=alpha,
           verbose=False,            
           valid_freq=0,
           timer=False,
           data_mask=[True,False])

t2 = time.time()
print(int(t2-t1)/60)


