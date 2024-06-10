
import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

from pinn      import PhysicsInformedNN
from equations import PME
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import copy
import time as time
from matplotlib.gridspec import GridSpec
########################################################################################################################################################################################################
#Parametros de la red

lr = 1e-4
layers  = [2] + 4*[32] + [1]

PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='tanh',
                         optimizer=keras.optimizers.legacy.Adam(lr),
                         #optimizer = 'lbfgs',
                         restore=True)
########################################################################################################################################################################################################
#Funciones 
def gif_sol(t):  
  filenames = []
  for i in t:    
    dom = np.array([(i,tt) for tt in np.linspace(np.min(-2),np.max(2),Nx)])
    #solution = convolution(dom)
    pinn = PINN.model(dom)[0]
    plt.title(f'Solucion a t = {i}')
    plt.plot(dom[:,1],pinn[:,0],label = 'PINN')
    #plt.plot(dom[:,1],solution,label = 'Solucion Real')
    plt.legend()
    # create file name and append it to a list
    filename = f'{i}.png'    
    for j in range(10):       
       filenames.append(filename)                 
    # save frame
    plt.savefig(filename)
    plt.close()  
  # build gif
  with imageio.get_writer('solution.gif', mode='I',duration=0.001) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)          
  # Remove files
  for filename in set(filenames):    
    os.remove(filename)    
def gaussian(x,s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )       
def gif_sol_rk2(x,t,s):  
  if len(t) != len(s):
     print('t and s dont have the same shape')
  filenames = []
  for i in range(len(t)):    
    dom = np.array([(t[i],tt) for tt in np.linspace(np.min(x),np.max(x),len(x))])
    plt.title(f'Solucion a t = {t[i]}')
    plt.plot(dom[:,1],s[i],label = 'RK2')
    plt.legend()
    # create file name and append it to a list
    filename = f'{i}.png'    
    for j in range(10):       
       filenames.append(filename)                 
    # save frame
    plt.savefig(filename)
    plt.close()  
  # build gif
  with imageio.get_writer('solution.gif', mode='I',duration=0.001) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)          
  # Remove files
  for filename in set(filenames):    
    os.remove(filename)     
def graph_space(Y,Nt,Nx,title):   
   solution = np.reshape(Y,(Nt,Nx))    
   for _ in range(3):
        solution = np.rot90(solution)
        solution = np.rot90(solution)
        solution = np.rot90(solution)
   plt.figure(figsize=(5,5))
   plt.title(title,fontsize=20)
   plt.imshow(solution, cmap = 'viridis',norm = 'linear',extent=[0,0.92,-2,2],aspect=0.17,vmin=0,vmax=np.max(Y)/4)
   plt.xticks(fontsize = 15)
   plt.yticks(fontsize = 15)
   cax = plt.axes([0.85, 0.1, 0.075, 0.8])
   plt.colorbar(cax=cax)
########################################################################################################################################################################################################
# Definicion de dominio, solucion real y PINN          
Nx = 100
Nt = 50

t = np.linspace(0,0.01,Nt)
x = np.linspace(-2,2,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

#Solucion real y solucion de la rex
fields = PINN.model(X)[0]
########################################################################################################################################################################################################
#Elijo los graficos que quiero ver 
loss = True # loss sola
cond_in = True # Condicion inicial
gif_sol(t)
########################################################################################################################################################################################################
t_ini = t[0]
x_eval_1 = np.array([(t_ini,tt) for tt in x])
u_eval_1 = gaussian(x_eval_1[:,1],0.5)
fields_eval_1 = PINN.model(x_eval_1)[0]

t_1 = t[40]
x_eval_3 = np.array([(t_1,tt) for tt in x])
fields_eval_3 = PINN.model(x_eval_3)[0]

#Derivadas parciales de la solucion de la red.
coords = tf.convert_to_tensor(x_eval_3)

with tf.GradientTape(persistent=True) as t2:
    t2.watch(coords)
    with tf.GradientTape(persistent=True) as t1:                
          t1.watch(coords)
          u = PINN.model(coords)[0]
          u2 = u * u

    grad_t = t1.gradient(u,coords)
    u_t = grad_t[:,0]
    grad_x = t1.gradient(u2,coords)

    del t1

grad_u2 = t2.gradient(grad_x,coords,unconnected_gradients=tf.UnconnectedGradients.ZERO)           
u_xx = grad_u2[:,1]                         

del t2 
########################################################################################################################################################################################################
# create objects
fig = plt.figure(figsize=(10,10))
gs = GridSpec(2, 2, figure=fig)
 
# create sub plots as grid
ax1 = fig.add_subplot(gs[1, :])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, : -1])


ax2.set_title('Residuales')
ax2.plot(x_eval_3[:,1],u_xx,label = r'$u_{xx}$')
ax2.plot(x_eval_3[:,1],u_t,label = r'$u_{t}$')
ax2.plot(x_eval_3[:,1],u_t + u_xx,label = r'$u_{t} + u_{xx} = 0$')
ax2.set_xlabel('x')
ax2.legend()

ax3.set_title('Condicion Inicial')
ax3.plot(x_eval_1[:,1],u_eval_1,label = 'Solution')
ax3.plot(x_eval_1[:,1],fields_eval_1,'x',label = 'PINN')  
ax3.set_xlabel('x')
ax3.set_ylabel('u(0,x)')
ax3.legend()

out = np.loadtxt('output.dat', unpack=True)
ax1.set_title('Training loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.semilogy(out[0], out[1],label='Loss data')
ax1.semilogy(out[0], out[2],label='Loss phys')
#ax1.semilogy(out[0], out[3],label='Loss bc')
ax1.legend()

# depict illustration
fig.suptitle("4*32")
######################################################################################################################################################################################################## 
if loss:
  plt.figure()
  out = np.loadtxt('output.dat', unpack=True)
  plt.semilogy(out[0], out[1],label='Loss data')
  plt.semilogy(out[0], out[2],label='Loss phys')
  plt.semilogy(out[0], out[3],label='Loss bc')
  plt.title('Training loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend() 
if cond_in:  
  plt.figure()
  plt.title(f'T = {t_ini}')
  plt.plot(x_eval_1[:,1],u_eval_1,label = 'Solution')
  plt.plot(x_eval_1[:,1],fields_eval_1,'x',label = 'PINN')  
  plt.xlabel('X')
  plt.ylabel('u(X)')
  plt.legend()

plt.show()

