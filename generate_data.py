import tensorflow as tf

a = tf.random.uniform([10000, 1000, 100, 150])
print(a.shape)
print(a[2,4,5,6])