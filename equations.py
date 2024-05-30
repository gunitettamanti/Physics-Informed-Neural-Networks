import tensorflow as tf
tf.keras.backend.set_floatx('float32')

@tf.function
def PME(model, coords, params):
    """ Opinion model
    Assumes coords is ordered in space and at constant t
    """            
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(coords)
        with tf.GradientTape(persistent=True) as t1:            
            t1.watch(coords)
            u = model(coords)[0]
            u2 = u * u
        
        grad_t = t1.gradient(u,coords)
        u_t = grad_t[:,0]
        grad_x = t1.gradient(u2,coords)

        del t1
        
    grad_2 = t2.gradient(grad_x,coords)
    u_xx = grad_2[:,1]
    
    del t2

    f = u_t + (u_xx)
    return [f]

