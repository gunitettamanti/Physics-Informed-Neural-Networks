""" PINN implementation of Porous Media Equation """
 
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float32')

from pinn      import PhysicsInformedNN
from equations import PME
import numpy as np
import time as time
import os
from distutils.dir_util import copy_tree
from shutil import copy
import json

def gaussian(x,s):
    return (1./np.sqrt( 2. * np.pi * s**2 )) * np.exp( -x**2 / ( 2. * s**2 ) )
def gen_domain(s,t0,t1,x,Nx,Nt,first_run):
    t = np.linspace(t0,t1,Nt)
    T,X = np.meshgrid(t,x)
    X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)
    Y = gaussian(X[:,1],s).reshape(X.shape[0],1)
    if first_run == False:
        cond_ini = np.loadtxt(f'{t[0]}.py', delimiter=',')
        Y[:Nx] = cond_ini.reshape(len(cond_ini),1)
    return X,Y

lr = [1e-4,1e-5]
layers  = [2] + 4*[32] + [1]

with open('params.json') as f:
    params = json.load(f)   

t0 = params['t0']
tfinal = params['tfinal']
x0 = params['x0']
x1 = params['x1']
Nx = params['Nx']
Nt = params['Nt']
delta_t = params['delta_t']
batch_size = params['batch_size']
s = params['s']

x_lins = np.linspace(x0,x1,20)
x_gauss = np.random.normal(0,s,Nx - len(x_lins))
x = np.sort(np.concatenate([x_lins,x_gauss]))

lambda_data = np.zeros(Nt*Nx) 
lambda_data[:Nx] = 1

lambda_phys = np.ones(Nt*Nx)*1e-1
lambda_phys[:Nx] = 0 #[0,1,1,..]

bc = np.zeros(Nx)
bc[:1] = 1
bc[-1:] = 1
lambda_bc = np.tile(bc,Nt)

flags = np.zeros(Nt*Nx)
flags[:Nx] = 1
eq_params = [0.01]

t1 = t0 + delta_t
X,Y = gen_domain(s,t0,t1,x,Nx,Nt,first_run=True)

PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='tanh', #probar con tanh 
                         norm_in=[np.array([t0,x0]), np.array([t1,x1])],                         
                         optimizer=keras.optimizers.Adam(lr[0]),
                         restore=True)
time1 = time.time()            
for lrate in lr:
    PINN.optimizer.learning_rate.assign(lrate)
    if lrate == lr[0]:
            tot_eps = 4500 
            PINN.train(X, 
               Y,
               PME,
               eq_params=eq_params,
               flags=flags,
               epochs=tot_eps,
               batch_size=batch_size,
               lambda_data=lambda_data,
               lambda_phys=lambda_phys,
               lambda_bc=lambda_bc
               )
    else:
        tot_eps = 2500
        PINN.train(X, 
               Y,
               PME,
               eq_params=eq_params,
               flags=flags,
               epochs=tot_eps,
               batch_size=batch_size,
               lambda_data=lambda_data,
               lambda_phys=lambda_phys,
               lambda_bc=lambda_bc
               )      
        

directory = f'{t0}-{t1}'

# Parent Directory path 
parent_dir = '/Users/gunitettamanti/Desktop/PME'

path = os.path.join(parent_dir, directory) 

os.mkdir(path)

dom = np.array([(t1, xi) for xi in X[:,1][:Nx]]).astype('float32')
cond_ini = PINN.model(dom)[0][:, 0]
np.savetxt(f'{t1}.py',cond_ini,delimiter = ',')

copy_tree('/Users/gunitettamanti/Desktop/PME/ckpt', f'/Users/gunitettamanti/Desktop/PME/{X[:,0][0]}-{X[:,0][-1]}/ckpt')
copy('/Users/gunitettamanti/Desktop/PME/output.dat', f'/Users/gunitettamanti/Desktop/PME/{X[:,0][0]}-{X[:,0][-1]}/output.dat')

times_txt = open("times.txt", "w")  
times_txt.write(f'{t0} {t1} ')
times_txt.close()

t0 = t0 + delta_t
t1 = t1 + delta_t

while delta_t > 0 or t1 < tfinal:

    print(f'{t0}-{t1}')
    X,Y = gen_domain(s,t0,t1,x,Nx,Nt,first_run=False)    

    PINN = PhysicsInformedNN(layers,
                         dest='./', 
                         activation='tanh', 
                         norm_in=[np.array([t0,x0]), np.array([t1,x1])],                         
                         optimizer=keras.optimizers.Adam(lr[0]),
                         restore=True)
    
    for lrate in lr:
        PINN.optimizer.learning_rate.assign(lrate)
        if lrate == lr[0]:
            tot_eps = 4500
            PINN.train(X, 
               Y,
               PME,
               eq_params=eq_params,
               flags=flags,
               epochs=tot_eps,
               batch_size=batch_size,
               lambda_data=lambda_data,
               lambda_phys=lambda_phys,
               lambda_bc=lambda_bc
               )
        else:
            tot_eps = 3000
            PINN.train(X, 
               Y,
               PME,
               eq_params=eq_params,
               flags=flags,
               epochs=tot_eps,
               batch_size=batch_size,
               lambda_data=lambda_data,
               lambda_phys=lambda_phys,
               lambda_bc=lambda_bc
               )      
        
    directory = f'{t0}-{t1}'

    # Parent Directory path 
    parent_dir = '/Users/gunitettamanti/Desktop/PME'

    path = os.path.join(parent_dir, directory) 

    os.mkdir(path)

    copy_tree('/Users/gunitettamanti/Desktop/PME/ckpt', f'/Users/gunitettamanti/Desktop/PME/{X[:,0][0]}-{X[:,0][-1]}/ckpt')
    copy('/Users/gunitettamanti/Desktop/PME/output.dat', f'/Users/gunitettamanti/Desktop/PME/{X[:,0][0]}-{X[:,0][-1]}/output.dat')
    
    u_t, u_xx = PME(PINN.model, X, [0.01], separate_terms=True)

    f = np.abs(u_t+0.01*u_xx)
    u_t_abs = np.abs(u_t)

    f = np.reshape(f,(Nt,Nx))
    u_t_abs = np.reshape(u_t_abs,(Nt,Nx))

    f_mean = np.mean(f,axis=1)
    u_t_mean = np.mean(u_t_abs,axis=1)

    crit = f_mean/u_t_mean
    print(crit)
    print(f_mean)
    print(u_t_mean)
    threshold = 0.1
    t_mod = 0.01
    if np.any(crit,where=crit>threshold):        
        delta_t -= t_mod    
        t1 = t1 - t_mod
        print('delta_t modified')
    else:
        with open('times.txt', 'a') as file:  
            file.write(f'{t0} {t1} ')
        os.remove(f"{t0}.py")
        t0 = t0 + delta_t
        t1 = t1 + delta_t
        dom = np.array([(t1, xi) for xi in X[:,1][:Nx]]).astype('float32')
        cond_ini = PINN.model(dom)[0][:, 0]
        np.savetxt(f'{t0}.py',cond_ini,delimiter = ',')

    if delta_t == 0:
        print('Condition fullfilled')
        break

os.remove(f"{t1}.py")
time2 = time.time()
print(int(time2-time1)/60)

