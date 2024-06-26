import tensorflow as tf
tf.keras.backend.set_floatx('float32')

@tf.function
def PME(model, coords, params, separate_terms=False):  
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(coords)
        with tf.GradientTape(persistent=True) as t1:            
            t1.watch(coords)
            u  = model(coords)[0][:, 0]
            u2 = u * u
        
        grad_t = t1.gradient(u,coords)
        u_t    = grad_t[:,0]

        grad_x = t1.gradient(u2,coords)
        u_x    = grad_x[:,1]

        
    grad_2 = t2.gradient(u_x,coords)
    u_xx = grad_2[:,1]
    
    del t1
    del t2

    if separate_terms:
        return u_t, u_xx
    else:
        f = u_t + 0.01*u_xx
        return [f]
