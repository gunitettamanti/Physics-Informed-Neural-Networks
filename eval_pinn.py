""" PINN implementation of opinion model """
"""TEST 1"""
########################################################################################################################################################################################################
#Librerias a utilizar

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.set_floatx('float64')

from pinn      import PhysicsInformedNN
from equations import opinion_model
import matplotlib.pyplot as plt
import numpy as np
import os
import imageio
import copy

########################################################################################################################################################################################################
#Parametros de la red

lr = 1e-5
layers  = [2] + 3*[64] + [2]

PINN = PhysicsInformedNN(layers,
                         dest='./', #saque el /odir porque no hacia falta 
                         activation='tanh',
                         optimizer=keras.optimizers.legacy.Adam(lr),
                         #optimizer = 'lbfgs',
                         restore=True)


########################################################################################################################################################################################################
#Funciones a utilizar

def gif_sol(t):  
  filenames = []
  for i in t:    
    dom = np.array([(i,tt) for tt in np.linspace(np.min(x),np.max(x),Nx)])
    solution = convolution(dom)
    pinn = PINN.model(dom)[0]
    plt.title(f'Solucion a t = {i}')
    plt.plot(dom[:,1],pinn[:,0],label = 'PINN')
    plt.plot(dom[:,1],solution,label = 'Solucion Real')
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
def gif_val(t):  
  filenames = []
  for i in t:    
    dom = np.array([(i,tt) for tt in np.linspace(np.min(x),np.max(x),Nx)])
    solution = convolution(dom)
    pinn = PINN.model(dom)[0][:,0]
    validation = ((pinn-solution)**2)/np.std(solution)
    plt.title(f'Solucion a t = {i}')    
    plt.plot(dom[:,1],validation,label = 'Error')
    plt.legend()
    # create file name and append it to a list
    filename = f'{i}.png'    
    for j in range(10):       
       filenames.append(filename)                 
    # save frame
    plt.savefig(filename)
    plt.close()  
  # build gif
  with imageio.get_writer('validation.gif', mode='I',duration=0.0000001) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)          
  # Remove files
  for filename in set(filenames):    
    os.remove(filename)    
def gaussian(x,s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )
def solution(X):  
  sol = np.zeros(len(X))
  for i in range(len(X)):
    x = X[i,1]
    t_0 = X[i,0]
    lower = t_0 - 1
    uper = 1 - t_0    
    sol[i] = np.where((x<lower) | (x>uper),0,1) * 1/(2-2*t_0) 
  return sol.reshape((len(X),1))
def convolution(X):
  s = 0.1
  t = len(np.unique(X[:,0]))  
  Nx = int(len(X)/t)  
  sol = np.zeros((Nx*t,1))
  for i in range(t):
    x_eval = X[i*Nx:(i+1)*Nx]          
    u_eval = solution(x_eval).reshape(-1)        
    gauss = gaussian(x_eval[:,1],s)  
    conv = np.convolve(u_eval,gauss,mode='same')  
    sol[i*Nx:(i+1)*Nx] = ((conv/np.max(conv))*np.max(u_eval)).reshape((len(u_eval),1))            
  return sol
def dif_fin(x,k):   
   dx = (x[:,1][1] - x[:,1][0])
   u = convolution(x).reshape(len(x[:,1]))
   u[1] = u[0]
   u[-2] = u[-1]
   u = u + k
   f = np.cumsum(u)*dx
   F = u*(2*f - 1)      
   return np.gradient(F, dx).reshape(len(x[:,1]),1)
def Euler(condicion_inicial,tiempo_final,n_pasos_temporales,espacio):
    h = tiempo_final/n_pasos_temporales
    tiempo = 0
    solucion = copy.copy(condicion_inicial)
    m = 0
    while m < n_pasos_temporales:    
        tiempo += h
        x = np.array([(tiempo,tt) for tt in espacio])
        F = dif_fin(x,0)        
        solucion += h*F
        m += 1
    return solucion
def dif_fin(x,k):   
   dx = (x[:,1][1] - x[:,1][0])
   u = convolution(x).reshape(len(x[:,1]))    
   u[1] = u[0]
   u[-2] = u[-1]  
   if type(k) != int:    
    u = u + k.reshape(len(x[:,1]))
   f = np.cumsum(u)*dx   
   F = u*(2*f - 1)      
   return np.gradient(F, dx).reshape(len(x[:,1]),1)
def RK2(condicion_inicial,tiempo_final,n_pasos_temporales,espacio):  
    h = tiempo_final/n_pasos_temporales
    tiempo = 0
    solucion = copy.copy(condicion_inicial)
    m = 0
    while m < n_pasos_temporales:    
        tiempo += h
        x1 = np.array([(tiempo,tt) for tt in espacio])
        F1 = dif_fin(x1,0)        
        k1 = h*F1        
        x2 = np.array([(tiempo + h,tt) for tt in espacio])
        F2 = dif_fin(x2,k1)
        k2 = h*F2
        solucion += 0.5 * (k1 + k2)
        m += 1
    return solucion    
def RK4(condicion_inicial,tiempo_final,n_pasos_temporales,espacio):  
    h = tiempo_final/n_pasos_temporales
    tiempo = 0
    solucion = copy.copy(condicion_inicial)
    m = 0
    while m < n_pasos_temporales:    
        tiempo += h
        x1 = np.array([(tiempo,tt) for tt in espacio])
        F1 = dif_fin(x1,0)        
        k1 = h*F1        
        x2 = np.array([(tiempo + h/2,tt) for tt in espacio])
        F2 = dif_fin(x2,k1/2)
        k2 = h*F2
        F3 = dif_fin(x2,k2/2)
        k3 = h*F3
        x4 = np.array([(tiempo + h,tt) for tt in espacio])
        F4 = dif_fin(x4,k3)
        k4 = h*F4
        solucion += (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        m += 1
    return solucion       
def gif_sol_rk2(x,t,s):  
  if len(t) != len(s):
     print('t and s dont have the same shape')
  filenames = []
  for i in range(len(t)):    
    dom = np.array([(t[i],tt) for tt in np.linspace(np.min(x),np.max(x),len(x))])
    solution = convolution(dom)
    plt.title(f'Solucion a t = {t[i]}')
    plt.plot(dom[:,1],s[i],label = 'RK2')
    plt.plot(dom[:,1],solution,label = 'Solucion Real')
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
  
Lx = 4 
Nx = 500
Nt = 50

t = np.linspace(0.93,0.94,Nt)
x = np.linspace(-2,2,Nx)

T,X = np.meshgrid(t,x)
X = np.hstack((np.sort(T.flatten()[:,None],axis=0),X.flatten(order='F')[:,None])) #Ordeno el vector como (t,x)

#Solucion real y solucion de la rex
Y = np.hstack((convolution(X), convolution(X))) 
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
loss = False # loss sola
cond_in = False # Condicion inicial
x_0 = False # Miro la solucion a un tiempo t_0
error_loc = False
loss_eq = False
Euler_graph = False
resi = False
mass = False
gif_sol(t)
#gif_val(t)


########################################################################################################################################################################################################
#Solucion final e inicial real y PINN

#Solucion a tiempo inicial.
""" t_ini = t[0]
x_eval_1 = np.array([(t_ini,tt) for tt in x])
u_eval_1 = convolution(x_eval_1)
fields_eval_1 = PINN.model(x_eval_1)[0]


#Solucion a tiempo final.
t_fijo = t[-1]
x_eval_2 = np.array([(t_fijo,tt) for tt in x])
u_eval_2 = convolution(x_eval_2)
fields_eval_2 = PINN.model(x_eval_2)[0] 

np.savetxt('0.97.py',fields_eval_2,delimiter = ',') """
#np.savetxt('0.05_D1+10.py',fields_eval_2,delimiter = ',')

########################################################################################################################################################################################################
#Validacion en el tiempo final
#val_local = ((u_eval_2-fields_eval_2)**2)/np.std(fields_eval_2)

########################################################################################################################################################################################################
#Calculo de derivadas y soluciones
""" 
dx = 4/Nx
#Integral a tiempo inicial.
fx = tf.cumsum(fields_eval_1[:,0])
fx_true = tf.cumsum(u_eval_1)
fx = fx*dx
fx_true = fx_true*dx 

I_pinn = fx[-1]
I_true = fx_true[-1]

#Integral a tiempo final.
fx_2 = tf.cumsum(fields_eval_2[:,0])
fx_true_2 = tf.cumsum(u_eval_2)
fx_2 = fx_2*dx
fx_true_2 = fx_true_2*dx 

I_pinn_2 = fx_2[-1]
I_true_2 = fx_true_2[-1]

#Derivadas parciales de la solucion de la red.
coords = tf.convert_to_tensor(x_eval_1)

with tf.GradientTape(persistent=True) as tape1:
      tape1.watch(coords)           
      Yp = PINN.model(coords)[0]
      u_p = Yp[:,0]
      fxp = Yp[:,1]      
      uf  = u_p*(2*fxp - 1)  

grad_u  = tape1.gradient(u_p, coords,unconnected_gradients=tf.UnconnectedGradients.ZERO)                         
u_t = grad_u[:,0] 
u_x = grad_u[:,1]                         
grad_uf = tape1.gradient(uf,coords,unconnected_gradients=tf.UnconnectedGradients.ZERO)           
uf_x = grad_uf[:,1]           
           
del tape1
        
f1 = uf_x
f2 = u_t 

#Evaluo el metodo de euler
# euler = Euler(u_eval_1,t[-1],Nt,x)
F = u_eval_1 * (2*fx_true  - 1)
F_pinn = fields_eval_1[:,0] * (2*fields_eval_1[:,1] - 1)  
F = F.numpy().reshape(len(F))
 """

########################################################################################################################################################################################################
#Runge Kutta Orden 2 

""" t_ini = t[0]
x_eval = np.array([(t_ini,tt) for tt in x])
cond_ini = convolution(x_eval)

t_final = t[-1]
x_eval_final = np.array([(t_final,tt) for tt in x])
sol_final = convolution(x_eval_final)

sol = []
for i in t:
    solution_rk2 = RK2(cond_ini,i,len(t),x)
    sol.append(solution_rk2)
sol_conc = np.concatenate(sol)
 """
#gif_sol_rk2(x,t,sol)

#graph_space(sol_conc,Nt,Nx,'RK2')
#graph_space(Y[:,0],Nt,Nx,'Solucion Real')

""" plt.figure(figsize=((20,20)))
plt.title(f'Solucion a t = {t[-1]}',size=20)  
plt.plot(x,sol[-1],'o',alpha = 0.8,label = 'Metodo RK2')    
plt.plot(x,sol_final,'o',alpha = 0.8,label = 'Solucion Real')    
plt.xlabel('X',size=15)
plt.ylabel('u',size=15)
plt.grid()
plt.legend(fontsize=15)
 """
#np.savetxt('RK2.py',sol_conc,delimiter = ',')

########################################################################################################################################################################################################
#Runge Kutta Orden 4 
""" 
sol = []
for i in t:
    solution_rk4 = RK4(cond_ini,i,len(t),x)
    sol.append(solution_rk2)
sol_conc = np.concatenate(sol)

gif_sol_rk2(x,t,sol)

graph_space(sol_conc,Nt,Nx,'RK4 ')
graph_space(Y[:,0],Nt,Nx,'Solucion Real')

plt.figure(figsize=((20,20)))
plt.title(f'Solucion a t = {t[-1]}',size=20)  
plt.plot(x,sol[-1],'o',alpha = 0.8,label = 'Metodo RK4')    
plt.plot(x,sol_final,'o',alpha = 0.8,label = 'Solucion Real')    
plt.xlabel('X',size=15)
plt.ylabel('u',size=15)
plt.grid()
plt.legend(fontsize=15)

np.savetxt('RK4.py',sol_conc,delimiter = ',')
 """
########################################################################################################################################################################################################
#Graficos


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
  plt.title(f'T = {t_ini}')
  plt.plot(x_eval_1[:,1],u_eval_1,label = 'Solution')
  plt.plot(x_eval_1[:,1],fields_eval_1[:,0],label = 'PINN')  
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


plt.show()

