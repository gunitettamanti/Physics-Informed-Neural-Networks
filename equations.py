import tensorflow as tf
tf.keras.backend.set_floatx('float32')

@tf.function
def opinion_model(model, coords, params):
    """ Opinion model
    Assumes coords is ordered in space and at constant t
    """
    # Calculate F(t_0, x)    
    dx = params[0]             
    Yp = model(coords)[0]      
    Fp = Yp[:,1]           
    split = tf.split(Yp[:,0],params[1],axis=0)
    s1 = tf.cumsum(split,axis=1)
    s2 = s1 * dx
    l = [s2[i] for i in range(len(s2))]
    Fx =  tf.concat(l,axis = 0)
    Ix_batch = [s2[i][-1] for i in range(len(s2))]    
    Ix_batch_t = [tf.ones_like(s2[0])*Ix_batch[i] for i in range(len(Ix_batch))]    
    Ix = tf.concat(Ix_batch_t,axis = 0)
        
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(coords)
        with tf.GradientTape(persistent=True) as t1:            
            t1.watch(coords)
            Yp = model(coords)[0]
        grad = t1.gradient(Yp, [coords[:,0],coords[:,1],coords[:,2],coords[:,3]])
        
    grad_2 = t2.gradient(grad,[coords[:,0],coords[:,1],coords[:,2],coords[:,3]])
    u_t = grad[:,0]
    u_xx = grad_2[:,1]
    u_yy = grad_2[:,2]
    u_zz = grad_2[:,3]

    f = u_t - (u_xx + u_yy + u_zz)
    return [f]

