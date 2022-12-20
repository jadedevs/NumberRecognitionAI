import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping
from data_process import x_train, y_train, x_test, y_test

tensorboard = TensorBoard(log_dir = './logs', histogram_freq = 1, profile_batch = 0, write_graph = True)
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3)  # stop training if the validation loss does not improve for 3 epochs

datagen = tf.keras.preprocessing.image.ImageDataGenerator( 
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = False
) # data augmentation

def create_model():
    model = tf.keras.Sequential( 
        [
            tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)), # convolutional layer, kernal 3x3, 32 filters
            tf.keras.layers.MaxPooling2D((2, 2)), # max pooling layer with size 2x2
            tf.keras.layers.Dropout(0.25), # add dropout regularization with rate 0.25

            tf.keras.layers.Conv2D(64, (3, 3), activation = 'relu'), # convolutional layer, kernal 3x3, 64 filters
            tf.keras.layers.MaxPooling2D((2, 2)), # max pooling layer with size 2x2
            tf.keras.layers.Dropout(0.25), # add dropout regularisation with rate 0.25

            tf.keras.layers.Flatten(), # flattens output of last layer
            tf.keras.layers.Dense(128, activation = 'relu'), # fully connected layer, 128 units, ReLU (rectified linear unit) activation
            tf.keras.layers.Dropout(0.5), # add dropout regularisation with rate 0.5
            tf.keras.layers.Dense(10, activation = 'softmax') # fully connected layer, 10 units, softmas activation
        ]
    ) # linear stack of layers

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


    return model

model = create_model()

model.fit(datagen.flow(x_train, y_train, batch_size = 32), epochs = 10, validation_data = (x_test, y_test), callbacks = [tensorboard, early_stopping])

test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
