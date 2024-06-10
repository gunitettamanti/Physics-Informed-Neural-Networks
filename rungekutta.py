''' Finite difference + RK solver for the porous media equation '''
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

def rhs(uu, dx):
    u2 = uu * uu
    du_dx   = np.gradient(u2, dx)
    d2u2_d2x = np.gradient(du_dx, dx)
    return -d2u2_d2x

def rk_step(uu, dx, dt, ord=2):
    ''' Variable-order Runge-Kutta time step '''
    up = np.copy(uu)

    for oo in range(ord, 0, -1):
        uu = up + dt*rhs(uu, dx)

        # Boundary conditions
        uu[1]  = uu[0]
        uu[-2] = uu[-1]

    return uu

# Initial condition
xx, dx = np.linspace(-2, 2, 100, retstep=True)
u0 = gaussian(xx,0.5)

# Time evolution
dt = 1e-3
Nt = int(0.015/dt)
tt = np.linspace(0, Nt*dt, Nt+1)
uu = [u0]
for _ in range(Nt):
    uu.append(rk_step(uu[-1], dx, dt))
uu = np.array(uu)

# Figures
# u(x,t)
plt.figure()
plt.imshow(uu.T, aspect='auto', origin='lower', extent=[0, Nt*dt, xx[0], xx[-1]])
plt.xlabel('x')
plt.ylabel('t')
plt.colorbar()

# u(x=0,t)
plt.figure()
plt.xlabel('t')
plt.ylabel('u')
plt.plot(tt, uu[:,50])

# u(x,t) at t=0, t=0.01, t=0.02
plt.figure()
plt.plot(xx, uu[0], label='t=0')
plt.plot(xx, uu[Nt//2], label='t=0.01')
plt.plot(xx, uu[-1], label='t=0.015')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()

plt.show()
