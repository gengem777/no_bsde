import tensorflow as tf

def tf_norm(a: tf.Tensor, b: tf.Tensor):
    assert tf.shape(a) == tf.shape(b)
    return tf.math.sqrt(tf.reduce_sum((a - b)**2))

def mc_price(x: tf.Tensor, u: tf.Tensor):
    """
    Returns the monte carlo price of a batch of asset paths and input function embedding.

            Parameters:
                    x (tf.float): a batch of underlying path shape: (batch, samples, time_steps, dim)
                    u (tf.float): a batch of imput function embedding vector shape: (batch, k)

            Returns:
                    mc_price (tf.float): a batch of mc_price
    """
    print(x.shape)
    assert len(tf.shape(u)) == 2
    assert len(tf.shape(x)) == 4
    # get the terminal value of x reduce the dimension to (batch, samples)
    x = tf.reduce_mean(x[:, :, -1, :], axis=-1)
    # get the discount factor
    df = tf.exp(u[:,0:1]) # (batch, 1)
    # get the mc paths
    mc_path = df * tf.nn.relu(x - u[:, -1][:,tf.newaxis])
    mc_price = tf.reduce_mean(mc_path, axis=1)
    return mc_price
