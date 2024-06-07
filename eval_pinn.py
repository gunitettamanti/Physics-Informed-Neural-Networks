""" PINN implementation of opinion model """
########################################################################################################################################################################################################
#Librerias a utilizar

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

lr = 1e-5
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
def dif_fin(sol,x,k):   
   u = sol * sol
   x = x[:,1].reshape(len(x[:,1]))
   u[1] = u[0]
   u[-2] = u[-1]
   if type(k) != int:    
    u = u + k.reshape(len(x)) 
   du_dx = np.gradient(u,x)
   d2u_d2x = np.gradient(du_dx,x)
   return -d2u_d2x
def RK2(condicion_inicial,tiempo_final,n_pasos_temporales,espacio):  
    solucion = copy.copy(condicion_inicial)
    h = tiempo_final/n_pasos_temporales
    tiempo = 0
    m = 0
    while m < n_pasos_temporales:    
        x1 = np.array([(tiempo,tt) for tt in espacio])
        F1 = dif_fin(solucion,x1,0)        
        k1 = h*F1        
        tiempo += h
        x2 = np.array([(tiempo + h,tt) for tt in espacio])
        F2 = dif_fin(solucion,x2,k1)
        k2 = h*F2
        solucion += 0.5 * (k1 + k2)
        m += 1
    return solucion           
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
Nt = 1000

t = np.linspace(0,0.1,Nt)
x = np.linspace(-2,2,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

#Solucion real y solucion de la rex
fields = PINN.model(X)[0]


#np.savetxt(f'{t[-1]}.py',fields,delimiter = ',')
########################################################################################################################################################################################################
#Elijo los graficos que quiero ver 
sol_3D_show = False
model_3D_show = False
model_show = False # Solucion de la red en todo el espacio 
sol_show = False # Solucion real en todo el espacio
loss_val = False # Funcion de perdida + Validation
val = False # Validation sola
loss = True # loss sola
cond_in = True # Condicion inicial
x_0 = False # Miro la solucion a un tiempo t_0
error_loc = False
loss_eq = False
Euler_graph = False
resi = False
mass = False
#gif_sol(t)
#gif_val(t)
########################################################################################################################################################################################################
t_ini = t[0]
x_eval_1 = np.array([(t_ini,tt) for tt in x])
u_eval_1 = gaussian(x_eval_1[:,1],0.5)
fields_eval_1 = PINN.model(x_eval_1)[0]

t_ini = t[25]
x_eval_2 = np.array([(t_ini,tt) for tt in x])
fields_eval_2 = PINN.model(x_eval_2)[0]

t_ini = t[50]
x_eval_3 = np.array([(t_ini,tt) for tt in x])
fields_eval_3 = PINN.model(x_eval_3)[0]

t_ini = t[75]
x_eval_4 = np.array([(t_ini,tt) for tt in x])
fields_eval_4 = PINN.model(x_eval_4)[0]


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
""" x_cond_ini = np.array([(0,tt) for tt in x])
cond_ini = gaussian(x_cond_ini[:,1],0.5)

tiempo_1 = time.time()
 
sol = []
for i in range(len(t)):
    solution_rk2 = RK2(cond_ini,t[i],i,x)
    sol.append(solution_rk2)

tiempo_2 = time.time()
print(tiempo_2-tiempo_1)
sol_conc = np.concatenate(sol)
gif_sol_rk2(x,t,sol) """
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
if sol_3D_show:
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0].flatten(), X[:,1].flatten(), convolution(X).flatten())
    ax.view_init(10,90)
if model_3D_show:  
    fig = plt.figure(figsize = (10,10))
    plt.title('Model')
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:,0].flatten(), X[:,1].flatten(), fields)
    ax.view_init(10,90)
if sol_show:
    solution = np.reshape(Y[:,0],(Nt,Nx))    
    for _ in range(3):
      solution = np.rot90(solution)
    plt.figure(figsize=(5,5))
    plt.imshow(solution, cmap = 'viridis',norm = 'linear',extent=[0,0.92,-2,2],aspect=0.17,vmin=0,vmax=np.max(Y)/4)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
if model_show:   
    model = np.reshape(fields,(Nt,Nx))    
    for _ in range(3):
      model = np.rot90(model)
    plt.figure(figsize=(5,5))
    plt.imshow(model, cmap = 'hot',extent=[0,np.max(t),t[0],t[-1]],aspect=0.4)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
if loss_val:
  fig, ax1 = plt.subplots()
  
  ax1.set_xlabel('Epochs')  
  ax1.set_ylabel('Loss', color='red')
  out = np.loadtxt('output.dat', unpack=True)
  lns1 = ax1.semilogy(out[0], out[2],color='red',label='Loss function')
  
  ax2 = ax1.twinx()  

  ax2.set_ylabel('Validation', color='blue')
  ax2.tick_params(axis='y', labelcolor='blue')  
  out_1 = np.loadtxt('validation.dat', unpack=True)
  lns2 = ax2.semilogy(out_1[0], out_1[1], color='blue',label='Validation')
  
  lns = lns1+lns2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc=0)        
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
if mass:
  out = np.loadtxt('mass_conservation.dat', unpack=True)
  plt.figure()
  plt.semilogy(out[0], out[1],label='Mass conservation')
  plt.title('Mass conservation')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
if val:
    plt.figure()  
    out_1 = np.loadtxt('validation.dat', unpack=True)
    plt.semilogy(out_1[0], out_1[1], label='Validation')          
    plt.title('Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')    
if cond_in:  
  plt.figure()
  #plt.title(f'T = {t_ini}')
  plt.plot(x_eval_1[:,1],u_eval_1,label = 'Solution')
  plt.plot(x_eval_1[:,1],fields_eval_1,'x',label = 'PINN')  
  plt.xlabel('X')
  plt.ylabel('u(X)')
  plt.grid()
  plt.legend()
if x_0:
  plt.figure()
  plt.title(f'Solucion a t = {t_fijo}')  
  plt.plot(x_eval_2[:,1],fields_eval_2[:,0], label = 'PINN')
  plt.plot(x_eval_2[:,1],u_eval_2, label = 'Solucion Real')    
  #plt.plot(x_eval_2[:,1],np.zeros(Nx),'o',alpha= 0.4,label = 'Collocation points')    
  plt.xlabel('X')
  plt.ylabel('$u(x,t = t_{0})$')
  plt.grid()
  plt.legend()
if error_loc:
   plt.figure()
   plt.title(f'Validacion a t = {t_fijo}')  
   plt.plot(x_eval_2[:,1],val_local, label = 'Error')   
   plt.xlabel('X')
   plt.ylabel('Error')
   plt.grid()
   plt.legend()
if Euler_graph:  
  plt.figure()
  plt.title(f'Solucion a t = {t_fijo}',size=20)  
  plt.plot(x_eval_2[:,1],euler,'o',alpha = 0.8,label = 'Metodo de Euler')    
  plt.plot(x_eval_2[:,1],u_eval_2, label = 'Solucion Real Convolucionada')   
  plt.xlabel('X',size=15)
  plt.ylabel('u',size=15)
  plt.grid()
  plt.legend(fontsize=15)
if resi:
  plt.figure()
  plt.title('Residuales a tiempo inicial')
  plt.plot(x_eval_1[:,1],u_t.numpy() - u_x.numpy().reshape(100)*(2*fx.numpy().reshape(100) - 1),label = '$u_{t} - uf_{x}$')
  plt.plot(x_eval_1[:,1],u_x.numpy().reshape(100)*(2*fx.numpy().reshape(100) - 1),label = '$uf_{x}$')
  plt.plot(x_eval_1[:,1],u_t.numpy(),label = '$u_{t}$')
  plt.legend()
  plt.grid()       

  plt.figure()
  plt.title('F')
  plt.plot(x_eval_1[:,1],fx_true,label = 'Integral real')
  plt.plot(x_eval_1[:,1],fxp,label = 'Integral PINN')
  plt.plot(x_eval_1[:,1],F,label = 'F real')
  plt.plot(x_eval_1[:,1],F_pinn,label = 'F PINN')
  plt.legend()
  plt.grid()   

  plt.figure()
  plt.title('$F_{x}$')
  plt.plot(x_eval_1[:,1],f1,label = '$F_{x}$ PINN')
  plt.plot(x_eval_1[:,1],np.diff(F,prepend = F[0])/dx,label='$F_{x}$ True')
  plt.legend()
  plt.grid()
  
  plt.figure()
  plt.title('$u_{t}$')
  plt.plot(x_eval_1[:,1],f2,label = '$u_{t}$ PINN')
  plt.plot(x_eval_1[:,1],dif_fin(x_eval_1),label = '$u_{t}$ Real')
  plt.legend()
  plt.grid()

  plt.figure()
  plt.title('Residuales')
  plt.plot(x_eval_1[:,1],f2,'o',label = '$u_{t}$ PINN') 
  plt.plot(x_eval_1[:,1],f1,label = '$F_{x}$ PINN')
  plt.plot(x_eval_1[:,1],f1 - f2,label = 'Residuos')
  plt.xlabel('X')
  plt.ylabel('$u_{t} - F_{x}$')
  plt.grid()
  plt.legend()

  plt.figure()
  plt.plot(x_eval_1[:,1],u_xx,label = r'$u_{xx}$')
  plt.plot(x_eval_1[:,1],u_t,label = r'$u_{t}$')
  plt.plot(x_eval_1[:,1],u_t + u_xx,label = r'$u_{t} + u_{xx} = 0$')
  plt.legend()



plt.show()

