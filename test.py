import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
#import tensorflow.keras import optimizers

#heigh = [170, 180, 175, 160]
#size = [260, 270, 265, 255]

키 = 170
신발 = 260

a = tf.Variable(0.1)
b = tf.Variable(0.2)

def diff():
    return tf.square(신발 - (키*a + b))

opt = tf.keras.optimizers.Adam(learning_rate=0.1)
for i in range(10):
    opt.minimize(diff, var_list = [a,b])
    print(a.numpy(), b.numpy())