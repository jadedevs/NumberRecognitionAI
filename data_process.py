import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalise data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# model expects grayscale 2D
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
