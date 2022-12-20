import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
import sys

from model import create_model

image_path = sys.argv[1]

image = imageio.imread(image_path)  # Load the image as a NumPy array

# Resize the image to the expected size and convert it to grayscale
image = np.array(image).astype('float32') / 255
image = np.mean(image, axis = -1, keepdims = True)  # Convert to grayscale
image = np.resize(image, (28, 28, 1))  # Resize to the expected size

# Expand the dimensions of the array to match the model's expected input shape
image_array = np.expand_dims(image, axis=0)

model = create_model()

predictions = model.predict(image_array)
predicted_label = np.argmax(predictions)
print(f'Predicted label: {predicted_label}')