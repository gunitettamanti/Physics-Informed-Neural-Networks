""" PINN implementation of Porous Media Equation"""
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import PME
import numpy as np
import time as time
import matplotlib.pyplot as plt

lr = 1e-5
layers  = [2] + 4*[32] + [1]

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

Nx = 100
Nt = 200

x = np.linspace(-2,2,Nx)
t = np.linspace(0,0.02,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)
Y = gaussian(X[:,1],0.5).reshape(X.shape[0],1)

#cond_ini = np.loadtxt('0.34.py', delimiter=',')
#Y[:Nx] = cond_ini[:,0].reshape(len(cond_ini),1)

lambda_data = np.zeros(Nt*Nx) #[1,0,0,..]
lambda_data[:Nx] = 1e2

lambda_phys = np.ones(Nt*Nx)
lambda_phys[:Nx] = 0 #[0,1,1,..]

bc = np.zeros(Nx)
bc[:1] = 1
bc[-1:] = 1
lambda_bc = np.tile(bc,Nt)

tot_eps = 10000
#PINN.validation = cte_validation(PINN,X,Y)

t1 = time.time()    
PINN.train(X, Y, PME,
           epochs=tot_eps,
           batch_size=Nt*Nx,    
           lambda_data=lambda_data,   # Punto donde se enfuerza L_bc
           lambda_phys=lambda_phys,
           lambda_bc=lambda_bc, 
           timer=False)
t2 = time.time()
print(int(t2-t1)/60)


